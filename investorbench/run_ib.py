#!/usr/bin/env python3
"""InvestorBench multi-asset extension for MemTrial (paid; gpt-4.1-mini-2025-04-14, temperature 0.7).

Benchmark: public InvestorBench data (Li et al., ACL 2025; data/SOURCE_COMMIT.txt), multi-asset mode. Four stocks with
complete prices (HON, JNJ, MSFT, UVV; NFLX is dropped because 41 of its warm-up/test days have no price) plus cash.
Official split: warm-up 2020-07-02..2020-09-30 (experience collection), test 2020-10-01..2021-05-05. Adapted to
portfolio allocation: every trading day the agent sets target weights over the four stocks and cash, held one day.

Phases (resumable: every API response is saved before use and replayed on later runs; nothing is paid twice):
  1 warm-up   one memory-free decision per warm-up day + one lesson written after its outcome -> experience bank
  2 embed     text-embedding-3-small vectors of the experiences and of each test day's context
  3 insights  ExpeL-style insights extracted once from the warm-up lessons
  4 test      per test day and seed: the 16 subsets of the 4 most similar experiences, and ExpeL (insights);
              per seed, sequential chains for FinMem-style, MemRL-style and Reflexion agents
Usage (from this folder; standard library only, python3 >= 3.8):
  python3 run_ib.py --mock      # no network: end-to-end test into run_mock/
  python3 run_ib.py --dry-run   # counts and cost estimate; no calls
  python3 run_ib.py --pilot     # a few real calls into run_pilot/
  python3 run_ib.py             # the full run into run/ (resumable)
Key: OPENAI_API_KEY, else the single OpenAI key in .env (never printed). Caps: pilot $1 (run_pilot/), main $25 (run/)."""
from __future__ import annotations
import argparse, hashlib, json, math, os, random, re, ssl, sys, threading, time, traceback, urllib.error, urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
CHAT_URL = "https://api.openai.com/v1/chat/completions"
EMB_URL = "https://api.openai.com/v1/embeddings"
MODEL, EMB_MODEL, TEMPERATURE = "gpt-4.1-mini-2025-04-14", "text-embedding-3-small", 0.7
P_IN, P_OUT, P_EMB = 0.40e-6, 1.60e-6, 0.02e-6
ASSETS = ["HON", "JNJ", "MSFT", "UVV"]; KEYS = ASSETS + ["CASH"]
WARM, TEST = ("2020-07-02", "2020-09-30"), ("2020-10-01", "2021-05-05")
SEEDS = [0, 1, 2]
K_RET, K_CAND = 4, 10                       # experiences per prompt; MemRL candidate pool
FINMEM_TAU, FINMEM_STEP = 90.0, 0.1         # FinMem-style recency decay (trading days) and importance step
MEMRL_LR, MEMRL_EPS = 0.2, 0.1              # MemRL-style value learning rate and exploration
REFLEXION_WINDOW = 3
MAX_DECISION_CALLS = 2                      # one retry only when the reply is not a valid weight vector


def now(): return datetime.now(timezone.utc).isoformat(timespec="seconds")
def sha(b): return hashlib.sha256(b if isinstance(b, bytes) else b.encode()).hexdigest()
def canon(v): return json.dumps(v, sort_keys=True, separators=(",", ":"))
def clean(s): return re.sub(r"\s+", " ", str(s)).strip()


# ----------------------------------------------------------------------------------------------- data and prompts
def load_data():
    D = {a: json.loads((HERE / "data" / f"{a.lower()}.json").read_text()) for a in ASSETS}
    days = sorted(set.intersection(*[set(d) for d in D.values()]))
    price = {a: {t: float(D[a][t]["prices"]) for t in days} for a in ASSETS}
    news = {a: {t: [clean(x) for x in (D[a][t].get("news") or [])] for t in days} for a in ASSETS}
    return days, price, news


class Bench:
    def __init__(self):
        self.days, self.price, self.news = load_data()
        self.idx = {t: i for i, t in enumerate(self.days)}
        self.warm = [t for t in self.days if WARM[0] <= t <= WARM[1]]
        self.test = [t for t in self.days if TEST[0] <= t <= TEST[1] and self.idx[t] + 1 < len(self.days)]

    def next_returns(self, t):
        i = self.idx[t]; n = self.days[i + 1]
        return {a: 100.0 * (self.price[a][n] / self.price[a][t] - 1.0) for a in ASSETS}

    def context(self, t):
        i = self.idx[t]
        L = [f"Date: {t}", "", "Market data (returns in %; volatility = standard deviation of daily returns over 20 days, in %):"]
        for a in ASSETS:
            p = self.price[a]; c = p[t]
            r = lambda k: 100.0 * (c / p[self.days[i - k]] - 1.0)
            dr = [100.0 * (p[self.days[j]] / p[self.days[j - 1]] - 1.0) for j in range(i - 19, i + 1)]
            mu = sum(dr) / 20.0; vol = math.sqrt(sum((x - mu) ** 2 for x in dr) / 19.0)
            L.append(f"{a}: close {c:.2f}; 1-day {r(1):+.2f}; 5-day {r(5):+.2f}; 20-day {r(20):+.2f}; volatility {vol:.2f}")
        L += ["", "News today:"]
        k = 0
        for a in ASSETS:
            for s in self.news[a][t][:3]:
                L.append(f"- {a}: {s[:240]}"); k += 1
        if k == 0: L.append("- (no news today)")
        return "\n".join(L)


SYSTEM = ("You are a portfolio manager. You allocate a long-only portfolio across four U.S. stocks (HON, JNJ, MSFT, UVV) "
          "and cash. Each trading day you set target weights that are held until the next trading day's close.")
ASK = ("\n\nDecide the target weights for the next trading day. Respond with a JSON object only, for example "
       '{"HON": 0.2, "JNJ": 0.2, "MSFT": 0.2, "UVV": 0.2, "CASH": 0.2, "reason": "one short sentence"}. '
       "Weights must be non-negative and sum to 1.")
FIX = "Your reply was not a valid JSON weight object. Reply again with the JSON object only, weights non-negative and summing to 1."


def block_experiences(texts):
    if not texts: return ""
    return "Experiences from your past decisions (use them only where relevant):\n" + "\n".join(f"[{i + 1}] {x}" for i, x in enumerate(texts))


def block_insights(ins):
    return "Insights distilled from your past decisions:\n" + "\n".join(f"{i + 1}. {x}" for i, x in enumerate(ins))


def block_reflections(refl):
    if not refl: return ""
    return "Your reflections on your most recent decisions (oldest first):\n" + "\n".join(f"- {x}" for x in refl)


def decision_body(ctx, block):
    user = ctx + (("\n\n" + block) if block else "") + ASK
    return {"model": MODEL, "temperature": TEMPERATURE, "max_tokens": 300,
            "messages": [{"role": "system", "content": SYSTEM}, {"role": "user", "content": user}]}


def fmt_w(w): return ", ".join(f"{k} {100 * w[k]:.0f}%" for k in KEYS)
def fmt_r(r): return ", ".join(f"{a} {r[a]:+.2f}" for a in ASSETS)
def port_ret(w, r): return sum(w[a] * r[a] for a in ASSETS)


def feedback_body(ctx, w, r, kind):
    pr, ew = port_ret(w, r), sum(r.values()) / 5.0
    task = ("Write one reusable lesson for future allocation decisions in at most 60 words: describe the market situation, "
            "what you did, what happened, and what you would do in a similar situation. Plain text only." if kind == "lesson" else
            "Reflect on this decision in at most 60 words: what worked, what did not, and what you will do differently "
            "next time. Plain text only.")
    user = (ctx + "\n\nYour decision: " + fmt_w(w) + "\nNext trading day returns (%): " + fmt_r(r) +
            f"\nYour portfolio return: {pr:+.2f}%; equal-weight portfolio (four stocks and cash): {ew:+.2f}%.\n\n" + task)
    return {"model": MODEL, "temperature": TEMPERATURE, "max_tokens": 160,
            "messages": [{"role": "system", "content": SYSTEM}, {"role": "user", "content": user}]}


def insight_body(exps):
    lines = [f"- [{e['date']}; portfolio {e['port']:+.2f}% vs equal weight {e['ew']:+.2f}%] {e['lesson']}" for e in exps]
    user = ("Below are lessons you wrote after past daily decisions, each with the outcome of that decision.\n\n" + "\n".join(lines) +
            "\n\nCompare the successful and unsuccessful decisions and extract at most 8 general insights for future daily "
            "allocation across HON, JNJ, MSFT, UVV and cash. Return a numbered list; each insight at most 30 words.")
    return {"model": MODEL, "temperature": TEMPERATURE, "max_tokens": 600,
            "messages": [{"role": "system", "content": SYSTEM}, {"role": "user", "content": user}]}


def parse_weights(text):
    m = re.search(r"\{.*\}", text or "", re.S)
    if not m: return None
    try: obj = json.loads(m.group(0))
    except Exception: return None
    try: w = {k: float(obj.get(k, 0.0)) for k in KEYS}
    except (TypeError, ValueError): return None
    if any(v < 0 or not math.isfinite(v) for v in w.values()): return None
    s = sum(w.values())
    if s <= 0: return None
    return {k: v / s for k, v in w.items()}, clean(obj.get("reason", ""))[:300]


def content(resp): return ((resp.get("choices") or [{}])[0].get("message") or {}).get("content") or ""


# ----------------------------------------------------------------------------------------------- API plumbing
def load_key():
    if os.environ.get("OPENAI_API_KEY"): return os.environ["OPENAI_API_KEY"]
    env = ROOT / ".env"
    if env.exists():
        keys = set(re.findall(r"sk-(?!or-)[A-Za-z0-9_-]{20,}", env.read_text()))
        if len(keys) == 1: return keys.pop()
    sys.exit("No OpenAI key: export OPENAI_API_KEY=... (or keep exactly one OpenAI key in .env)")


def ssl_context():
    try:
        import certifi
        return ssl.create_default_context(cafile=certifi.where())
    except ImportError:
        return ssl.create_default_context()


class BudgetStop(RuntimeError): pass


class Ledger:
    def __init__(self, cap, out):
        self.cap, self.lock, self.spent, self.held = cap, threading.Lock(), 0.0, 0.0
        for p in (out / "calls").glob("*.json"): self.spent += json.loads(p.read_text())["charged_usd"]
        fp = out / "call_failures.jsonl"
        if fp.exists(): self.spent += sum(json.loads(x)["charged_usd"] for x in fp.read_text().splitlines() if x.strip())

    def reserve(self, amt):
        with self.lock:
            if self.spent + self.held + amt > self.cap + 1e-12: return False
            self.held += amt; return True

    def settle(self, held, charged):
        with self.lock: self.held -= held; self.spent += charged


def mock_response(url, body, logical_id):
    rnd = random.Random(sha(logical_id))
    if url == EMB_URL:
        vecs = []
        for x in body["input"]:
            r2 = random.Random(sha(x)); v = [r2.gauss(0, 1) for _ in range(32)]; n = math.sqrt(sum(a * a for a in v))
            vecs.append([a / n for a in v])
        return {"data": [{"embedding": v} for v in vecs], "usage": {"prompt_tokens": 10 * len(vecs)}}
    user = body["messages"][-1]["content"]
    if "JSON object" in user and "Decide the target weights" in user or user.startswith("Your reply was not"):
        a = [rnd.random() for _ in KEYS]; s = sum(a)
        obj = {k: round(v / s, 3) for k, v in zip(KEYS, a)}; obj["reason"] = "mock"; txt = json.dumps(obj)
        if rnd.random() < 0.03: txt = "not json"
    elif "numbered list" in user:
        txt = "\n".join(f"{i}. Mock insight {i}: diversify when volatility rises." for i in range(1, 6))
    else:
        txt = f"Mock note {rnd.randint(0, 999)}: in mixed news, hold more cash and trim the most volatile stock."
    return {"choices": [{"message": {"content": txt}}], "usage": {"prompt_tokens": 1000, "completion_tokens": 80}}


def make_requester(ledger, out, key, ctx, stop, mock):
    def requester(logical_id, body, url=CHAT_URL):
        wire = canon(body).encode(); rsha = sha(wire)
        path = out / "calls" / (sha(logical_id)[:40] + ".json")
        if path.exists():
            rec = json.loads(path.read_text())
            if rec["request_sha256"] != rsha: raise RuntimeError(f"saved call {logical_id} has a different request")
            return rec["response"]
        if stop.is_set(): raise BudgetStop("stopped")
        reserve = (len(wire) / 3.0 * P_EMB) if url == EMB_URL else (len(wire) / 3.0 * P_IN + body.get("max_tokens", 1000) * P_OUT)
        if not ledger.reserve(reserve): stop.set(); raise BudgetStop("hard cap reached; not sending")
        charged, last = 0.0, None
        try:
            for attempt in range(1, 7):
                if mock: resp = mock_response(url, body, logical_id)
                else:
                    req = urllib.request.Request(url, data=json.dumps(body).encode(), headers={
                        "Authorization": f"Bearer {key}", "Content-Type": "application/json"})
                    try:
                        with urllib.request.urlopen(req, timeout=180, context=ctx) as r: resp = json.loads(r.read())
                    except urllib.error.HTTPError as e:
                        last = f"HTTP {e.code}: {e.read()[:300].decode(errors='replace')}"
                        if e.code in (401, 403): stop.set(); raise RuntimeError(last)
                        if e.code == 429 or e.code >= 500:
                            try: wait = float(e.headers.get("retry-after") or 0)
                            except ValueError: wait = 0.0
                            time.sleep(min(max(wait, 2.0 ** attempt), 60)); continue
                        raise RuntimeError(last)
                    except (urllib.error.URLError, TimeoutError, ConnectionError, OSError) as e:
                        last = f"{type(e).__name__}: {e}"; time.sleep(min(2.0 ** attempt, 30)); continue
                u = resp.get("usage") or {}
                if url == EMB_URL: charged = u.get("prompt_tokens", 0) * P_EMB if u else reserve
                else: charged = (u.get("prompt_tokens", 0) * P_IN + u.get("completion_tokens", 0) * P_OUT) if u else reserve
                rec = {"logical_id": logical_id, "request_sha256": rsha, "charged_usd": charged, "saved_utc": now(),
                       "request": body, "response": resp}
                tmp = path.with_suffix(".tmp"); tmp.write_text(json.dumps(rec)); os.replace(tmp, path)
                return resp
            raise RuntimeError(f"gave up after retries: {last}")
        except Exception as e:
            if not path.exists():
                with ledger.lock, (out / "call_failures.jsonl").open("a") as f:
                    f.write(json.dumps({"logical_id": logical_id, "error": str(e)[:300], "charged_usd": charged, "utc": now()}) + "\n")
            raise
        finally:
            ledger.settle(reserve, charged)
    return requester


# ----------------------------------------------------------------------------------------------- one decision
def decide(req, lid, ctx, block):
    """at most MAX_DECISION_CALLS calls; returns (weights or None, reason, calls)."""
    body = decision_body(ctx, block); resp = req(lid + "|c1", body); txt = content(resp); p = parse_weights(txt)
    if p: return p[0], p[1], 1
    body2 = dict(body); body2["messages"] = body["messages"] + [{"role": "assistant", "content": txt}, {"role": "user", "content": FIX}]
    p = parse_weights(content(req(lid + "|c2", body2)))
    return (p[0], p[1], 2) if p else (None, "invalid reply twice", 2)


def save_exec(out, rec):
    path = out / "executions" / (sha(rec["key"])[:40] + ".json")
    tmp = path.with_suffix(".tmp"); tmp.write_text(json.dumps(rec)); os.replace(tmp, path)


def exec_exists(out, key): return (out / "executions" / (sha(key)[:40] + ".json")).exists()


def cos(u, v): return sum(a * b for a, b in zip(u, v))


# ----------------------------------------------------------------------------------------------- phases
def phase_warmup(B, req, out, warm_days, workers):
    def one(t):
        ctx = B.context(t); w, reason, n = decide(req, f"warm|{t}", ctx, "")
        if w is None: return {"date": t, "status": "FAILED"}
        r = B.next_returns(t)
        lesson = clean(content(req(f"warm-lesson|{t}", feedback_body(ctx, w, r, "lesson"))))[:700]
        rec = {"key": f"warm|{t}", "phase": "warmup", "date": t, "weights": w, "reason": reason, "lesson": lesson,
               "port": port_ret(w, r), "ew": sum(r.values()) / 5.0, "status": "VALID", "calls": n + 1}
        save_exec(out, rec); return rec
    with ThreadPoolExecutor(workers) as ex: recs = list(ex.map(one, warm_days))
    exps = [{"id": f"E{i:03d}", "date": r["date"], "lesson": r["lesson"], "port": r["port"], "ew": r["ew"],
             "text": f"({r['date']}) {r['lesson']}"} for i, r in enumerate(x for x in recs if x["status"] == "VALID")]
    (out / "experiences.json").write_text(json.dumps(exps, indent=1)); return exps


def phase_embed(B, req, out, exps, test_days):
    texts = [("exp", e["id"], e["text"]) for e in exps] + [("ctx", t, B.context(t)) for t in test_days]
    vec = {}
    for i in range(0, len(texts), 64):
        chunk = texts[i:i + 64]
        resp = req(f"emb|{sha(canon([c[2] for c in chunk]))[:24]}", {"model": EMB_MODEL, "input": [c[2] for c in chunk]}, url=EMB_URL)
        for c, d in zip(chunk, resp["data"]):
            v = d["embedding"]; n = math.sqrt(sum(a * a for a in v)) or 1.0; vec[(c[0], c[1])] = [a / n for a in v]
    E = {"model": EMB_MODEL, "experiences": {k[1]: v for k, v in vec.items() if k[0] == "exp"},
         "contexts": {k[1]: v for k, v in vec.items() if k[0] == "ctx"}}
    (out / "embeddings.json").write_text(json.dumps(E)); return E


def phase_insights(req, out, exps):
    txt = content(req("expel-insights", insight_body(exps)))
    ins = [clean(re.sub(r"^\s*\d+[\.\)]\s*", "", x)) for x in txt.splitlines() if re.match(r"^\s*\d+[\.\)]", x)][:8]
    if not ins: ins = [clean(txt)[:300]]
    (out / "insights.json").write_text(json.dumps(ins, indent=1)); return ins


def retrieval(E, exps, t, k):
    c = E["contexts"][t]
    return sorted((e["id"] for e in exps), key=lambda h: (-cos(c, E["experiences"][h]), h))[:k]


def run_job(B, req, out, job):
    key = job["key"]
    if exec_exists(out, key): return "skip"
    ctx = B.context(job["date"])
    try:
        w, reason, n = decide(req, key, ctx, job["block"])
        rec = {k: v for k, v in job.items() if k != "block"}
        rec.update(phase="test", weights=w, reason=reason, calls=n, status="VALID" if w else "FAILED")
        save_exec(out, rec); return rec["status"]
    except BudgetStop: return "budget"


def chain(B, req, out, kind, seed, test_days, exps, E, log):
    """sequential agent whose memory depends on its own earlier outcomes (FinMem-style, MemRL-style, Reflexion)."""
    text = {e["id"]: e["text"] for e in exps}; edate = {e["id"]: e["date"] for e in exps}
    imp = {e["id"]: 0.5 for e in exps}; Q = {e["id"]: 0.0 for e in exps}; refl = []
    rnd = random.Random(f"{kind}|{seed}")
    for t in test_days:
        key = f"{kind}|{t}|{seed}"; ctx = B.context(t); c = E["contexts"][t]
        if kind == "finmem":
            age = lambda h: B.idx[t] - B.idx[edate[h]]
            score = {h: cos(c, E["experiences"][h]) + math.exp(-age(h) / FINMEM_TAU) + imp[h] for h in text}
            used = sorted(text, key=lambda h: (-score[h], h))[:K_RET]; block = block_experiences([text[h] for h in used])
        elif kind == "memrl":
            cand = retrieval(E, exps, t, K_CAND)
            explore = rnd.random() < MEMRL_EPS; pick = rnd.sample(cand, K_RET)
            used = pick if explore else sorted(cand, key=lambda h: (-Q[h], cand.index(h)))[:K_RET]
            block = block_experiences([text[h] for h in used])
        else:
            used = []; block = block_reflections(refl[-REFLEXION_WINDOW:])
        try:
            w, reason, n = decide(req, key, ctx, block)
        except BudgetStop:
            log.append(f"{kind} seed {seed}: budget stop at {t}"); return
        rec = {"key": key, "phase": "test", "arm": kind, "date": t, "seed": seed, "mask": None, "used": used,
               "weights": w, "reason": reason, "calls": n, "status": "VALID" if w else "FAILED"}
        if w is None:
            save_exec(out, rec); continue                     # no outcome: memory state unchanged
        r = B.next_returns(t); pr = port_ret(w, r)
        if kind == "finmem":
            for h in used: imp[h] = min(1.0, max(0.0, imp[h] + (FINMEM_STEP if pr > 0 else -FINMEM_STEP)))
        elif kind == "memrl":
            for h in used: Q[h] += MEMRL_LR * (pr - Q[h])
        else:
            try: note = clean(content(req(f"reflexion-note|{t}|{seed}", feedback_body(ctx, w, r, "reflection"))))[:500]
            except BudgetStop:
                save_exec(out, rec); log.append(f"reflexion seed {seed}: budget stop at {t}"); return
            refl.append(note); rec["reflection"] = note; rec["calls"] += 1
        save_exec(out, rec)


def plan_jobs(test_days, exps, E, ins, seeds):
    jobs = []
    for t in test_days:
        top = retrieval(E, exps, t, K_RET); text = {e["id"]: e["text"] for e in exps}
        for s in seeds:
            for mask in range(16):
                used = [top[j] for j in range(K_RET) if (mask >> j) & 1]
                jobs.append({"key": f"subset|{t}|{s}|{mask}", "arm": "subset", "date": t, "seed": s, "mask": mask,
                             "top4": top, "used": used, "block": block_experiences([text[h] for h in used])})
            jobs.append({"key": f"expel|{t}|{s}", "arm": "expel", "date": t, "seed": s, "mask": None, "used": [],
                         "block": block_insights(ins)})
    return jobs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mock", action="store_true"); ap.add_argument("--pilot", action="store_true"); ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--workers", type=int, default=8)
    a = ap.parse_args()
    B = Bench()
    if a.pilot:
        out, cap, warm_days, test_days, seeds = HERE / "run_pilot", 1.0, B.warm[:6], B.test[:2], [0]
    elif a.mock:
        out, cap, warm_days, test_days, seeds = HERE / "run_mock", 1e9, B.warm, B.test[:12], SEEDS
    else:
        out, cap, warm_days, test_days, seeds = HERE / "run", 25.0, B.warm, B.test, SEEDS
    n_dec = len(warm_days) + len(test_days) * len(seeds) * (16 + 1 + 3)
    n_fb = len(warm_days) + len(test_days) * len(seeds)
    est = n_dec * (1700 * P_IN + 120 * P_OUT) + n_fb * (1500 * P_IN + 90 * P_OUT) + 7000 * P_IN + 400 * P_OUT
    print(f"warm-up days {len(warm_days)}, test days {len(test_days)} ({test_days[0]}..{test_days[-1]}), seeds {seeds}; "
          f"decisions {n_dec}, lessons/reflections {n_fb}; estimated cost ${est:.2f}; cap ${cap:.2f}; folder {out.name}", flush=True)
    if a.dry_run:
        print(B.context(test_days[0])); return
    (out / "calls").mkdir(parents=True, exist_ok=True); (out / "executions").mkdir(exist_ok=True)
    ledger = Ledger(cap, out); stop = threading.Event()
    req = make_requester(ledger, out, None if a.mock else load_key(), ssl_context(), stop, a.mock)
    t0 = time.time()
    try:
        exps = phase_warmup(B, req, out, warm_days, a.workers)
        print(f"warm-up: {len(exps)} experiences; spent ${ledger.spent:.3f} ({time.time() - t0:.0f}s)", flush=True)
        E = phase_embed(B, req, out, exps, test_days); ins = phase_insights(req, out, exps)
        print(f"embeddings {len(E['experiences'])} + {len(E['contexts'])}; insights {len(ins)}; spent ${ledger.spent:.3f}", flush=True)
    except BudgetStop:
        print("budget stop during setup"); return
    jobs = plan_jobs(test_days, exps, E, ins, seeds); log = []
    threads = [threading.Thread(target=chain, args=(B, req, out, k, s, test_days, exps, E, log), daemon=True)
               for k in ("finmem", "memrl", "reflexion") for s in seeds]
    for th in threads: th.start()
    status = {}
    with ThreadPoolExecutor(a.workers) as ex:
        futs = [ex.submit(run_job, B, req, out, j) for j in jobs]
        for i, f in enumerate(as_completed(futs)):
            try: st = f.result()
            except Exception as e: st = "error"; log.append(f"job error: {type(e).__name__}: {str(e)[:200]}")
            status[st] = status.get(st, 0) + 1
            if (i + 1) % 500 == 0: print(f"  {i + 1}/{len(jobs)} jobs; spent ${ledger.spent:.3f}; {time.time() - t0:.0f}s", flush=True)
    for th in threads: th.join()
    n_exec = len(list((out / "executions").glob("*.json")))
    summ = {"jobs": status, "executions_saved": n_exec, "spent_usd_conservative": round(ledger.spent, 6), "cap_usd": cap,
            "seconds": round(time.time() - t0, 1), "finished_utc": now(), "log": log[:50], "stopped": stop.is_set()}
    (out / "RUN_SUMMARY.json").write_text(json.dumps(summ, indent=1)); print(json.dumps(summ, indent=1))


if __name__ == "__main__":
    main()
