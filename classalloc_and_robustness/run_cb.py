#!/usr/bin/env python3
"""ClassAlloc: the third real benchmark (paid; by default gpt-4.1-mini-2025-04-14 at temperature 0.7, as the main runs).

Task. Every month, on the first trading session, the agent sets long-only target weights over five asset classes
(equities, bonds, commodities, real estate, cash), held for 20 sessions. Prices: PortBench's public raw data, one
equal-weighted index per class (classalloc/data.json, built by cb_build_data.py). The prompt shows the date and, for
each class, the 1-, 3- and 6-month returns and the 20-day volatility. Decisions follow each other, so holdings are
carried over and the 15 bp fee is charged on what is traded (cb_eval.py), as on InvestorBench.

Protocol (as InvestorBench, run_ib.py):
  1 warm-up   58 months (Jul 2015 - Apr 2020): one memory-free decision per month and one lesson written by the agent
              after seeing its 20-session outcome -> a fixed pool of 58 lessons
  2 embed     text-embedding-3-small vectors of the lessons and of each test month's context
  3 insights  ExpeL-style insights extracted once from the warm-up lessons
  4 test      71 months (May 2020 - Mar 2026; 21 of them from Jul 2024 on, after the knowledge cutoff of gpt-4.1-mini):
              per month and seed, the 16 subsets of the 4 most similar lessons and ExpeL; per seed, sequential chains for
              FinMem-style, MemRL-style and Reflexion agents, each of which learns the outcome of a decision once its
              20-session window has ended (in 9 of the 70 test months the next decision comes after 19 sessions, and
              that outcome is learned one decision later). Seeds 0, 1, 2.
Usage (from this folder):
  python3 run_cb.py --mock        # no network: end-to-end test into classalloc/run_mock
  python3 run_cb.py --dry-run     # counts, cost estimate and one prompt; no calls
  python3 run_cb.py --probe       # one real call (< USD 0.01)
  python3 run_cb.py               # full run into classalloc/run (resumable; hard cap USD 6)
Options: --model NAME --temperature T (see llm_client.MODELS) write into classalloc/run_<model>_t<T> instead.
"""
from __future__ import annotations
import argparse, json, math, random, re, sys, threading, time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

HERE = Path(__file__).resolve().parent; sys.path.insert(0, str(HERE))
import llm_client as LC
from llm_client import clean, content, canon, sha

DATA = HERE / "classalloc" / "data.json"
CLASSES = ["equities", "bonds", "commodities", "real_estate", "cash"]
LABEL = {"equities": "equities", "bonds": "bonds", "commodities": "commodities", "real_estate": "real estate", "cash": "cash"}
SEEDS = [0, 1, 2]
K_RET, K_CAND = 4, 10
FINMEM_TAU, FINMEM_STEP = 4.5, 0.1          # FinMem-style recency decay (months, about 90 trading days) and importance step
MEMRL_LR, MEMRL_EPS = 0.2, 0.1
REFLEXION_WINDOW = 3
BASE_MODEL, BASE_T = "gpt-4.1-mini", 0.7


class Bench:
    def __init__(self):
        D = json.loads(DATA.read_text()); assert D["classes"] == CLASSES, "data.json was built for other classes"
        self.sess, self.L = D["sessions"], D["level"]; self.sidx = {s: i for i, s in enumerate(self.sess)}
        self.dates, self.warm, self.test, self.hold = D["decision_dates"], D["warmup"], D["test"], D["hold_sessions"]
        self.month = {t: i for i, t in enumerate(self.dates)}

    def ret(self, c, i, k): return 100.0 * (self.L[c][i] / self.L[c][i - k] - 1.0)

    def context(self, t):
        i = self.sidx[t]
        L = [f"Date: {t}", "", "Market data (each asset class and cash is an equal-weighted index of exchange-traded assets; "
             "returns in %; volatility = standard deviation of daily returns over the last 20 trading days, in %):"]
        for c in CLASSES:
            dr = [self.ret(c, j, 1) for j in range(i - 19, i + 1)]; mu = sum(dr) / 20.0
            vol = math.sqrt(sum((x - mu) ** 2 for x in dr) / 19.0)
            L.append(f"{LABEL[c]}: 1-month {self.ret(c, i, 21):+.2f}; 3-month {self.ret(c, i, 63):+.2f}; "
                     f"6-month {self.ret(c, i, 126):+.2f}; volatility {vol:.2f}")
        return "\n".join(L)

    def outcome(self, t):
        """class returns over the 20 holding sessions, in %."""
        i = self.sidx[t]; return {c: 100.0 * (self.L[c][i + self.hold] / self.L[c][i] - 1.0) for c in CLASSES}


SYSTEM = ("You are a portfolio manager. You allocate a long-only portfolio across four asset classes (equities, bonds, "
          "commodities, real estate) and cash. At the start of each month you set target weights that are held for the "
          "next 20 trading days.")
ASK = ("\n\nDecide the target weights for the next 20 trading days. Respond with a JSON object only, for example "
       '{"equities": 0.2, "bonds": 0.2, "commodities": 0.2, "real_estate": 0.2, "cash": 0.2, "reason": "one short sentence"}. '
       "Weights must be non-negative and sum to 1.")
FIX = "Your reply was not a valid JSON weight object. Reply again with the JSON object only, weights non-negative and summing to 1."
MAX_DECISION_CALLS = 2


def block_experiences(texts):
    if not texts: return ""
    return "Experiences from your past decisions (use them only where relevant):\n" + "\n".join(f"[{i + 1}] {x}" for i, x in enumerate(texts))


def block_insights(ins): return "Insights distilled from your past decisions:\n" + "\n".join(f"{i + 1}. {x}" for i, x in enumerate(ins))


def block_reflections(refl):
    if not refl: return ""
    return "Your reflections on your most recent decisions (oldest first):\n" + "\n".join(f"- {x}" for x in refl)


def fmt_w(w): return ", ".join(f"{LABEL[k]} {100 * w[k]:.0f}%" for k in CLASSES)
def fmt_r(r): return ", ".join(f"{LABEL[c]} {r[c]:+.2f}" for c in CLASSES)
def port_ret(w, r): return sum(w[c] * r[c] for c in CLASSES)


class Bodies:
    """chat bodies for one model / temperature (prompt text fixed)."""
    def __init__(self, spec, temperature): self.spec, self.T = spec, temperature

    def _b(self, user, max_tokens):
        return LC.adapt({"model": "", "temperature": 0.0, "max_tokens": max_tokens,
                         "messages": [{"role": "system", "content": SYSTEM}, {"role": "user", "content": user}]}, self.spec, self.T)

    def decision(self, ctx, block): return self._b(ctx + (("\n\n" + block) if block else "") + ASK, 300)

    def feedback(self, ctx, w, r, kind):
        ew = sum(r.values()) / len(CLASSES)
        task = ("Write one reusable lesson for future allocation decisions in at most 60 words: describe the market situation, "
                "what you did, what happened, and what you would do in a similar situation. Plain text only." if kind == "lesson" else
                "Reflect on this decision in at most 60 words: what worked, what did not, and what you will do differently "
                "next time. Plain text only.")
        user = (ctx + "\n\nYour decision: " + fmt_w(w) + "\nReturns over the next 20 trading days (%): " + fmt_r(r) +
                f"\nYour portfolio return: {port_ret(w, r):+.2f}%; equal-weight portfolio (the five asset classes): {ew:+.2f}%.\n\n" + task)
        return self._b(user, 160)

    def insights(self, exps):
        lines = [f"- [{e['date']}; portfolio {e['port']:+.2f}% vs equal weight {e['ew']:+.2f}%] {e['lesson']}" for e in exps]
        user = ("Below are lessons you wrote after past monthly decisions, each with the outcome of that decision.\n\n" + "\n".join(lines) +
                "\n\nCompare the successful and unsuccessful decisions and extract at most 8 general insights for future monthly "
                "allocation across equities, bonds, commodities, real estate and cash. Return a numbered list; each insight at most 30 words.")
        return self._b(user, 600)


def parse_weights(text):
    m = re.search(r"\{.*\}", text or "", re.S)
    if not m: return None
    try: obj = json.loads(m.group(0))
    except Exception: return None
    try: w = {k: float(obj.get(k, obj.get(LABEL[k], 0.0))) for k in CLASSES}
    except (TypeError, ValueError): return None
    if any(v < 0 or not math.isfinite(v) for v in w.values()): return None
    s = sum(w.values())
    if s <= 0: return None
    return {k: v / s for k, v in w.items()}, clean(obj.get("reason", ""))[:300]


def decide(req, BD, lid, ctx, block):
    body = BD.decision(ctx, block); resp = req(lid + "|c1", body); txt = content(resp); p = parse_weights(txt)
    if p: return p[0], p[1], 1
    body2 = json.loads(json.dumps(body)); body2["messages"] += [{"role": "assistant", "content": txt}, {"role": "user", "content": FIX}]
    p = parse_weights(content(req(lid + "|c2", body2)))
    return (p[0], p[1], 2) if p else (None, "invalid reply twice", 2)


def save_exec(out, rec):
    path = out / "executions" / (sha(rec["key"])[:40] + ".json")
    tmp = path.with_suffix(".tmp"); tmp.write_text(json.dumps(rec)); tmp.replace(path)


def exec_exists(out, key): return (out / "executions" / (sha(key)[:40] + ".json")).exists()
def cos(u, v): return sum(a * b for a, b in zip(u, v))


def phase_warmup(B, BD, req, out, warm, workers):
    def one(t):
        ctx = B.context(t); w, reason, n = decide(req, BD, f"warm|{t}", ctx, "")
        if w is None: return {"date": t, "status": "FAILED"}
        r = B.outcome(t)
        lesson = clean(content(req(f"warm-lesson|{t}", BD.feedback(ctx, w, r, "lesson"))))[:700]
        rec = {"key": f"warm|{t}", "phase": "warmup", "date": t, "weights": w, "reason": reason, "lesson": lesson,
               "port": port_ret(w, r), "ew": sum(r.values()) / len(CLASSES), "status": "VALID", "calls": n + 1}
        save_exec(out, rec); return rec
    with ThreadPoolExecutor(workers) as ex: recs = list(ex.map(one, warm))
    exps = [{"id": f"E{i:03d}", "date": r["date"], "lesson": r["lesson"], "port": r["port"], "ew": r["ew"],
             "text": f"({r['date']}) {r['lesson']}"} for i, r in enumerate(x for x in recs if x["status"] == "VALID")]
    (out / "experiences.json").write_text(json.dumps(exps, indent=1)); return exps


def phase_embed(B, req, out, exps, test):
    texts = [("exp", e["id"], e["text"]) for e in exps] + [("ctx", t, B.context(t)) for t in test]
    vec = {}
    for i in range(0, len(texts), 64):
        chunk = texts[i:i + 64]
        resp = req(f"emb|{sha(canon([c[2] for c in chunk]))[:24]}", {"model": LC.EMB_MODEL, "input": [c[2] for c in chunk]}, kind="emb")
        for c, d in zip(chunk, resp["data"]):
            v = d["embedding"]; n = math.sqrt(sum(a * a for a in v)) or 1.0; vec[(c[0], c[1])] = [a / n for a in v]
    E = {"model": LC.EMB_MODEL, "experiences": {k[1]: v for k, v in vec.items() if k[0] == "exp"},
         "contexts": {k[1]: v for k, v in vec.items() if k[0] == "ctx"}}
    (out / "embeddings.json").write_text(json.dumps(E)); return E


def phase_insights(BD, req, out, exps):
    txt = content(req("expel-insights", BD.insights(exps)))
    ins = [clean(re.sub(r"^\s*\d+[\.\)]\s*", "", x)) for x in txt.splitlines() if re.match(r"^\s*\d+[\.\)]", x)][:8]
    if not ins: ins = [clean(txt)[:300]]
    (out / "insights.json").write_text(json.dumps(ins, indent=1)); return ins


def retrieval(E, exps, t, k):
    c = E["contexts"][t]
    return sorted((e["id"] for e in exps), key=lambda h: (-cos(c, E["experiences"][h]), h))[:k]


def plan_jobs(test, exps, E, ins, seeds):
    jobs = []; text = {e["id"]: e["text"] for e in exps}
    for t in test:
        top = retrieval(E, exps, t, K_RET)
        for s in seeds:
            for mask in range(16):
                used = [top[j] for j in range(K_RET) if (mask >> j) & 1]
                jobs.append({"key": f"subset|{t}|{s}|{mask}", "arm": "subset", "date": t, "seed": s, "mask": mask,
                             "top4": top, "used": used, "block": block_experiences([text[h] for h in used])})
            jobs.append({"key": f"expel|{t}|{s}", "arm": "expel", "date": t, "seed": s, "mask": None, "used": [],
                         "block": block_insights(ins)})
    return jobs


def run_job(B, BD, req, out, job):
    key = job["key"]
    if exec_exists(out, key): return "skip"
    try:
        w, reason, n = decide(req, BD, key, B.context(job["date"]), job["block"])
        rec = {k: v for k, v in job.items() if k != "block"}
        rec.update(phase="test", weights=w, reason=reason, calls=n, status="VALID" if w else "FAILED")
        save_exec(out, rec); return rec["status"]
    except LC.BudgetStop: return "budget"


def chain(B, BD, req, out, kind, seed, test, exps, E, log):
    """sequential agents whose memory depends on their own earlier outcomes (as run_ib.chain; recency in months).
    The outcome of a decision enters the agent's memory once its holding window of B.hold sessions has ended."""
    text = {e["id"]: e["text"] for e in exps}; edate = {e["id"]: e["date"] for e in exps}
    imp = {e["id"]: 0.5 for e in exps}; Q = {e["id"]: 0.0 for e in exps}; refl = []; pend = []
    rnd = random.Random(f"{kind}|{seed}")

    def learn(used, w, r, note):
        pr = port_ret(w, r)
        if kind == "finmem":
            for h in used: imp[h] = min(1.0, max(0.0, imp[h] + (FINMEM_STEP if pr > 0 else -FINMEM_STEP)))
        elif kind == "memrl":
            for h in used: Q[h] += MEMRL_LR * (pr - Q[h])
        else: refl.append(note)

    for t in test:
        key = f"{kind}|{t}|{seed}"; ctx = B.context(t); c = E["contexts"][t]
        while pend and pend[0][0] <= B.sidx[t]: learn(*pend.pop(0)[1])
        if kind == "finmem":
            age = lambda h: B.month[t] - B.month[edate[h]]
            score = {h: cos(c, E["experiences"][h]) + math.exp(-age(h) / FINMEM_TAU) + imp[h] for h in text}
            used = sorted(text, key=lambda h: (-score[h], h))[:K_RET]; block = block_experiences([text[h] for h in used])
        elif kind == "memrl":
            cand = retrieval(E, exps, t, K_CAND)
            explore = rnd.random() < MEMRL_EPS; pick = rnd.sample(cand, K_RET)
            used = pick if explore else sorted(cand, key=lambda h: (-Q[h], cand.index(h)))[:K_RET]
            block = block_experiences([text[h] for h in used])
        else:
            used = []; block = block_reflections(refl[-REFLEXION_WINDOW:])
        if exec_exists(out, key):                         # resume: replay the saved decision to rebuild the memory state
            rec = json.loads((out / "executions" / (sha(key)[:40] + ".json")).read_text()); w = rec["weights"]
            if w is None: continue
        else:
            try: w, reason, n = decide(req, BD, key, ctx, block)
            except LC.BudgetStop:
                log.append(f"{kind} seed {seed}: budget stop at {t}"); return
            rec = {"key": key, "phase": "test", "arm": kind, "date": t, "seed": seed, "mask": None, "used": used,
                   "weights": w, "reason": reason, "calls": n, "status": "VALID" if w else "FAILED"}
            if w is None:
                save_exec(out, rec); continue
        r = B.outcome(t); note = None
        if kind == "reflexion":
            if "reflection" in rec: note = rec["reflection"]
            else:
                try: note = clean(content(req(f"reflexion-note|{t}|{seed}", BD.feedback(ctx, w, r, "reflection"))))[:500]
                except LC.BudgetStop:
                    save_exec(out, rec); log.append(f"reflexion seed {seed}: budget stop at {t}"); return
                rec["reflection"] = note; rec["calls"] = rec.get("calls", 1) + 1
        save_exec(out, rec)
        pend.append((B.sidx[t] + B.hold, (used, w, r, note)))      # learned once the holding window has ended


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mock", action="store_true"); ap.add_argument("--dry-run", action="store_true"); ap.add_argument("--probe", action="store_true")
    ap.add_argument("--model", default=BASE_MODEL); ap.add_argument("--temperature", type=float, default=BASE_T)
    ap.add_argument("--cap", type=float, default=6.0); ap.add_argument("--workers", type=int, default=8)
    a = ap.parse_args()
    spec = LC.MODELS[a.model]; T = None if spec["reasoning"] else a.temperature; BD = Bodies(spec, T); B = Bench()
    tag = "run" if (a.model, a.temperature) == (BASE_MODEL, BASE_T) else f"run_{a.model}_t{a.temperature}"
    if a.mock: tag = "run_mock"
    out = HERE / "classalloc" / tag
    warm, test, seeds = B.warm, (B.test[:12] if a.mock else B.test), SEEDS
    n_dec = len(warm) + len(test) * len(seeds) * (16 + 1 + 3); n_fb = len(warm) + len(test) * len(seeds)
    p_dec = len(canon(BD.decision(B.context(test[0]), block_experiences(["x" * 420] * 4)))) / 3.6
    est = n_dec * (p_dec * spec["p_in"] + 80 * spec["p_out"]) + n_fb * (900 * spec["p_in"] + 90 * spec["p_out"])
    print(f"ClassAlloc: model {spec['model']} (temperature {T}); warm-up {len(warm)} months ({warm[0]}..{warm[-1]}), "
          f"test {len(test)} months ({test[0]}..{test[-1]}), seeds {seeds}; decisions {n_dec}, lessons/reflections {n_fb}; "
          f"estimated cost ${est:.2f}; cap ${a.cap:.2f}; folder {out}", flush=True)
    if a.dry_run:
        print("\n--- example prompt (test month 1, no memory) ---\n" + SYSTEM + "\n\n" + B.context(test[0]) + ASK); return
    if a.probe:
        po = HERE / "classalloc" / "probe"; (po / "calls").mkdir(parents=True, exist_ok=True)
        stop = threading.Event(); led = LC.Ledger(0.05, po); req = LC.make_requester(led, po, spec, stop, False); t0 = time.time()
        w, reason, n = decide(req, BD, f"probe|{test[0]}|{time.time():.0f}", B.context(test[0]), "")
        print(f"probe: weights {w}; reason {reason!r}; calls {n}; {time.time() - t0:.1f}s; cost ${led.spent:.5f}"); return
    (out / "calls").mkdir(parents=True, exist_ok=True); (out / "executions").mkdir(exist_ok=True)
    man = {"benchmark": "ClassAlloc", "model": spec["model"], "temperature": T, "cap_usd": a.cap, "seeds": seeds,
           "warmup": [warm[0], warm[-1], len(warm)], "test": [test[0], test[-1], len(test)],
           "data_sha256": sha(DATA.read_bytes()), "run_cb_sha256": sha(Path(__file__).read_bytes()), "started_utc": LC.now(),
           "feedback": "each outcome is learned once its holding window has ended"}
    if not (out / "MANIFEST.json").exists(): (out / "MANIFEST.json").write_text(json.dumps(man, indent=1))
    ledger = LC.Ledger(a.cap, out); stop = threading.Event(); req = LC.make_requester(ledger, out, spec, stop, a.mock); t0 = time.time()
    try:
        exps = phase_warmup(B, BD, req, out, warm, a.workers)
        print(f"warm-up: {len(exps)} lessons; spent ${ledger.spent:.3f} ({time.time() - t0:.0f}s)", flush=True)
        E = phase_embed(B, req, out, exps, test); ins = phase_insights(BD, req, out, exps)
        print(f"embeddings {len(E['experiences'])} + {len(E['contexts'])}; insights {len(ins)}; spent ${ledger.spent:.3f}", flush=True)
    except LC.BudgetStop:
        print("budget stop during setup"); return
    jobs = plan_jobs(test, exps, E, ins, seeds); log = []
    threads = [threading.Thread(target=chain, args=(B, BD, req, out, k, s, test, exps, E, log), daemon=True)
               for k in ("finmem", "memrl", "reflexion") for s in seeds]
    for th in threads: th.start()
    status = {}
    with ThreadPoolExecutor(a.workers) as ex:
        futs = [ex.submit(run_job, B, BD, req, out, j) for j in jobs]
        for i, f in enumerate(as_completed(futs)):
            try: st = f.result()
            except Exception as e: st = "error"; log.append(f"job error: {type(e).__name__}: {str(e)[:200]}")
            status[st] = status.get(st, 0) + 1
            if (i + 1) % 500 == 0: print(f"  {i + 1}/{len(jobs)} jobs; spent ${ledger.spent:.3f}; {time.time() - t0:.0f}s", flush=True)
    for th in threads: th.join()
    summ = {"jobs": status, "executions_saved": len(list((out / "executions").glob("*.json"))),
            "spent_usd_conservative": round(ledger.spent, 6), "cap_usd": a.cap, "seconds": round(time.time() - t0, 1),
            "finished_utc": LC.now(), "log": log[:50], "stopped": stop.is_set()}
    (out / "RUN_SUMMARY.json").write_text(json.dumps(summ, indent=1)); print(json.dumps(summ, indent=1))


if __name__ == "__main__":
    main()
