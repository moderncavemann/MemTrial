#!/usr/bin/env python3
"""ClassAlloc: replay the FinMem-style, MemRL-style and Reflexion chains with each outcome released only once its
20-session holding window has ended (paid).

ClassAlloc decides on the first trading session of every month and scores each decision over the next 20 sessions.
In 9 of the 70 test months the next decision comes 19 sessions later, one session before the previous outcome is
complete, and the chains of the logged runs received that outcome at once. This script replays every chain with the
outcome released after its window. Where the prompt of a decision is unchanged, the saved decision is kept (it is a draw for
exactly that request); where it changes, the decision, and for Reflexion the note written after it, is drawn anew.
All other decisions of the run (the 16 subsets and ExpeL) do not depend on outcomes and are copied unchanged.

This is how the ClassAlloc runs of the paper were brought to the protocol that run_cb.py now implements directly; it needs the
raw logs of a run (classalloc/<run>/executions), which will be released with the paper.

Usage (from this folder):
  python3 run_cb_delayfix.py --dry-run [--model M] [--masked]   # counts and cost estimate; no calls
  python3 run_cb_delayfix.py [--model M] [--masked]             # writes classalloc/<run>_dfix (resumable; cap USD 3)
The run folder is classalloc/run (gpt-4.1-mini), classalloc/run_<M>_t0.7 or, with --masked, classalloc/run_masked-<M>_t0.7.
"""
from __future__ import annotations
import argparse, json, math, random, shutil, sys, threading, time
from pathlib import Path

HERE = Path(__file__).resolve().parent; sys.path.insert(0, str(HERE))
import llm_client as LC
from llm_client import clean, content, sha
import run_cb as RC
KINDS = ("finmem", "memrl", "reflexion")


def tag_of(a):
    if a.mock: return "run_masked_mock" if a.masked else "run_mock"
    if a.masked: return f"run_masked-{a.model}_t{a.temperature}"
    return "run" if (a.model, a.temperature) == (RC.BASE_MODEL, RC.BASE_T) else f"run_{a.model}_t{a.temperature}"


def load_execs(run):
    recs = {}
    for f in (run / "executions").glob("*.json"):
        r = json.loads(f.read_text()); recs[r["key"]] = r
    return recs


def chain_fixed(B, BD, req, out, kind, seed, test, exps, E, orig, dry, res, lock, log):
    text = {e["id"]: e["text"] for e in exps}; edate = {e["id"]: e["date"] for e in exps}
    new = lambda: {"imp": {h: 0.5 for h in text}, "Q": {h: 0.0 for h in text}, "refl": [], "rnd": random.Random(f"{kind}|{seed}")}
    so, sf = new(), new()                    # so: original chain (outcome at once), sf: fixed chain (outcome after its window)
    pend = []; kept = drawn = 0; first = None

    def inputs(st, t):
        c = E["contexts"][t]
        if kind == "finmem":
            age = lambda h: B.month[t] - B.month[edate[h]]
            score = {h: RC.cos(c, E["experiences"][h]) + math.exp(-age(h) / RC.FINMEM_TAU) + st["imp"][h] for h in text}
            used = sorted(text, key=lambda h: (-score[h], h))[:RC.K_RET]
            return used, RC.block_experiences([text[h] for h in used])
        if kind == "memrl":
            cand = RC.retrieval(E, exps, t, RC.K_CAND)
            explore = st["rnd"].random() < RC.MEMRL_EPS; pick = st["rnd"].sample(cand, RC.K_RET)
            used = pick if explore else sorted(cand, key=lambda h: (-st["Q"][h], cand.index(h)))[:RC.K_RET]
            return used, RC.block_experiences([text[h] for h in used])
        return [], RC.block_reflections(st["refl"][-RC.REFLEXION_WINDOW:])

    def update(st, used, w, r, note):          # exactly the update of run_cb.chain
        pr = RC.port_ret(w, r)
        if kind == "finmem":
            for h in used: st["imp"][h] = min(1.0, max(0.0, st["imp"][h] + (RC.FINMEM_STEP if pr > 0 else -RC.FINMEM_STEP)))
        elif kind == "memrl":
            for h in used: st["Q"][h] += RC.MEMRL_LR * (pr - st["Q"][h])
        else: st["refl"].append(note)

    for t in test:
        key = f"{kind}|{t}|{seed}"; ctx = B.context(t); i = B.sidx[t]; r = B.outcome(t)
        while pend and pend[0][0] <= i: update(sf, *pend.pop(0)[1])
        o_used, o_block = inputs(so, t); f_used, f_block = inputs(sf, t)
        orec = orig.get(key)
        if orec is None: raise RuntimeError(f"no saved decision {key}")
        if kind != "reflexion" and orec.get("used") != o_used: raise RuntimeError(f"replay of the saved chain differs at {key}")
        if orec.get("weights") is not None: update(so, o_used, orec["weights"], r, orec.get("reflection"))
        if f_block == o_block:
            rec = dict(orec); rec["delayfix"] = "kept"; kept += 1; w = rec["weights"]; note = rec.get("reflection")
        else:
            drawn += 1; first = first or t
            if dry:
                rec = None; w = orec["weights"]; note = orec.get("reflection")     # stand-in to continue the count
            else:
                path = out / "executions" / (sha(key)[:40] + ".json")
                if path.exists(): rec = json.loads(path.read_text())             # resume
                else:
                    try: w, reason, n = RC.decide(req, BD, key + "|dfix", ctx, f_block)
                    except LC.BudgetStop:
                        log.append(f"{kind} seed {seed}: budget stop at {t}"); return
                    rec = {"key": key, "phase": "test", "arm": kind, "date": t, "seed": seed, "mask": None, "used": f_used,
                           "weights": w, "reason": reason, "calls": n, "status": "VALID" if w else "FAILED", "delayfix": "drawn"}
                    if w is not None and kind == "reflexion":
                        try: rec["reflection"] = clean(content(req(f"reflexion-note|{t}|{seed}|dfix", BD.feedback(ctx, w, r, "reflection"))))[:500]
                        except LC.BudgetStop:
                            log.append(f"{kind} seed {seed}: budget stop at {t} (note)"); return
                        rec["calls"] += 1
                w = rec["weights"]; note = rec.get("reflection")
        if rec is not None and not dry: RC.save_exec(out, rec)
        if w is None: continue                                                   # as run_cb.chain: no update after a failed decision
        pend.append((i + B.hold, (f_used, w, r, note)))
    with lock: res[f"{kind}|{seed}"] = {"kept": kept, "drawn": drawn, "first_change": first,
                                       "upper_bound": sum(1 for t in test if first and t >= first)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mock", action="store_true"); ap.add_argument("--dry-run", action="store_true"); ap.add_argument("--masked", action="store_true")
    ap.add_argument("--model", default=RC.BASE_MODEL); ap.add_argument("--temperature", type=float, default=RC.BASE_T)
    ap.add_argument("--cap", type=float, default=3.0)
    a = ap.parse_args()
    spec = LC.MODELS[a.model]; T = None if spec["reasoning"] else a.temperature
    if a.masked:
        import run_cb_masked as RM; B, BD = RM.MaskedBench(), RM.MaskedBodies(spec, T)
    else:
        B, BD = RC.Bench(), RC.Bodies(spec, T)
    src = HERE / "classalloc" / tag_of(a); dst = HERE / "classalloc" / (tag_of(a) + "_dfix")
    if json.loads((src / "MANIFEST.json").read_text()).get("feedback"):
        print(f"{src.name} was made by a run_cb.py that already learns each outcome after its window; nothing to replay"); return
    exps = json.loads((src / "experiences.json").read_text()); E = json.loads((src / "embeddings.json").read_text())
    orig = load_execs(src); seeds = RC.SEEDS
    test = sorted({r["date"] for r in orig.values() if r.get("arm") in KINDS})
    short = [(x, y) for x, y in zip(test[:-1], test[1:]) if B.sidx[y] - B.sidx[x] < B.hold]
    print(f"{src.name}: {len(test)} test months; {len(short)} of {len(test) - 1} months follow the previous decision "
          f"before its {B.hold}-session window ends", flush=True)
    res, lock, log = {}, threading.Lock(), []
    if a.dry_run:
        for k in KINDS:
            for s in seeds: chain_fixed(B, BD, None, None, k, s, test, exps, E, orig, True, res, lock, log)
        dr = sum(v["drawn"] for v in res.values()); ub = sum(v["upper_bound"] for v in res.values())
        notes = sum(v["upper_bound"] for k, v in res.items() if k.startswith("reflexion"))
        p_dec = len(LC.canon(BD.decision(B.context(test[0]), RC.block_experiences(["x" * 420] * 4)))) / 3.6
        est = (ub + notes) * (p_dec * spec["p_in"] + 90 * spec["p_out"])
        print(json.dumps(res, indent=1)); print(f"decisions that change at once: {dr}; at most {ub} decisions and {notes} notes "
                                                f"to draw (every later decision of a changed chain); estimated cost at most ${est:.2f}"); return
    (dst / "calls").mkdir(parents=True, exist_ok=True); (dst / "executions").mkdir(exist_ok=True)
    for f in ("experiences.json", "embeddings.json", "insights.json"):
        if (src / f).exists() and not (dst / f).exists(): shutil.copy2(src / f, dst / f)
    for r in orig.values():                                  # decisions that do not depend on outcomes, copied unchanged
        if r.get("arm") not in KINDS:
            p = dst / "executions" / (sha(r["key"])[:40] + ".json")
            if not p.exists(): p.write_text(json.dumps(r))
    man = {"benchmark": "ClassAlloc", "source_run": src.name, "dates_hidden": a.masked, "model": spec["model"], "temperature": T,
           "protocol": f"each outcome is released to the FinMem-style, MemRL-style and Reflexion agents only after its {B.hold}-session window",
           "cap_usd": a.cap, "run_cb_sha256": sha(Path(RC.__file__).read_bytes()), "delayfix_sha256": sha(Path(__file__).read_bytes()),
           "started_utc": LC.now()}
    if not (dst / "MANIFEST.json").exists(): (dst / "MANIFEST.json").write_text(json.dumps(man, indent=1))
    ledger = LC.Ledger(a.cap, dst); stop = threading.Event(); req = LC.make_requester(ledger, dst, spec, stop, a.mock); t0 = time.time()
    th = [threading.Thread(target=chain_fixed, args=(B, BD, req, dst, k, s, test, exps, E, orig, False, res, lock, log), daemon=True)
          for k in KINDS for s in seeds]
    for x in th: x.start()
    for x in th: x.join()
    summ = {"chains": res, "spent_usd_conservative": round(ledger.spent, 6), "cap_usd": a.cap, "seconds": round(time.time() - t0, 1),
            "finished_utc": LC.now(), "log": log[:50], "stopped": stop.is_set(),
            "complete": len(res) == len(KINDS) * len(seeds) and not log}
    (dst / "RUN_SUMMARY.json").write_text(json.dumps(summ, indent=1)); print(json.dumps(summ, indent=1))


if __name__ == "__main__":
    main()
