#!/usr/bin/env python3
"""Self-consistency with three seeds: two more memory-free decisions per test date and seed (paid; small).

The published Self-consistency baseline averages the three seeds' memory-free drafts of a date, so it is one portfolio per
date and its sd over seeds is 0 by construction. This script adds two more memory-free decisions per test date and seed
(draws 1 and 2; draw 0 is the logged one), with the prompt, model and temperature of the run being extended, so that seed s
of Self-consistency can average three draws of its own. Outputs go to sc_runs/<bench>_<run>/ only (published runs are not
touched). Resumable: saved calls are replayed at no cost; a hard cap stops each run.

  python3 run_sc_extra.py --bench ib --run main|t0.0|t0.3|t1.0|nano|gpt5mini|llama70b [--mock] [--cap X]
  python3 run_sc_extra.py --bench cb --run main|gpt-4.1-nano|gpt-5-mini|llama-3.3-70b [--mock] [--cap X]
  (PortBench: the extra drafts are part of ../portbench/data/executions.jsonl.gz, source "sc".)
"""
from __future__ import annotations
import argparse, json, sys, threading, time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
HERE = Path(__file__).resolve().parent; LAB = HERE.parent
sys.path.insert(0, str(HERE)); sys.path.insert(0, str(LAB / "investorbench")); sys.path.insert(0, str(LAB / "portbench"))
SEEDS, DRAWS = [0, 1, 2], [1, 2]


def run_pool(jobs, one, workers, label, spent):
    status = {}; t0 = time.time()
    with ThreadPoolExecutor(workers) as ex:
        futs = [ex.submit(one, j) for j in jobs]
        for i, f in enumerate(as_completed(futs)):
            try: st = f.result()
            except Exception as e: st = f"error {type(e).__name__}: {str(e)[:120]}"
            status[st] = status.get(st, 0) + 1
            if (i + 1) % 200 == 0: print(f"  {label}: {i + 1}/{len(jobs)}; spent ${spent():.3f}; {time.time() - t0:.0f}s", flush=True)
    return status, time.time() - t0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bench", required=True, choices=["ib", "cb"]); ap.add_argument("--run", default="main")
    ap.add_argument("--mock", action="store_true"); ap.add_argument("--check", action="store_true")
    ap.add_argument("--cap", type=float, default=None); ap.add_argument("--workers", type=int, default=8)
    a = ap.parse_args()
    out = HERE / "sc_runs" / (f"{a.bench}_{a.run}" + ("_mock" if a.mock else "")); stop = threading.Event()
    if a.bench in ("ib", "cb"):
        (out / "calls").mkdir(parents=True, exist_ok=True); (out / "executions").mkdir(exist_ok=True)
        import llm_client as LC
        if a.bench == "ib":
            import run_ib as RB
            if a.run == "main":
                cap = a.cap or 1.5; ledger = RB.Ledger(cap, out)
                req = RB.make_requester(ledger, out, None if a.mock else RB.load_key(), RB.ssl_context(), stop, a.mock); Stop = RB.BudgetStop
            else:
                import run_ib_variant as RV
                V = RV.VARIANTS[a.run]; spec = LC.MODELS[V["model"]]; RV.patch_bodies(spec, V["temperature"])
                cap = a.cap or 1.5; ledger = LC.Ledger(cap, out); req = LC.make_requester(ledger, out, spec, stop, a.mock); Stop = RB.BudgetStop
            B = RB.Bench(); days = B.test[:12] if a.mock else B.test
            decide = lambda key, t: RB.decide(req, key, B.context(t), ""); save, exists = RB.save_exec, RB.exec_exists
        else:
            import run_cb as RC
            model = RC.BASE_MODEL if a.run == "main" else a.run; spec = LC.MODELS[model]
            T = None if spec["reasoning"] else RC.BASE_T; BD = RC.Bodies(spec, T); B = RC.Bench(); days = B.test[:12] if a.mock else B.test
            cap = a.cap or 1.0; ledger = LC.Ledger(cap, out); req = LC.make_requester(ledger, out, spec, stop, a.mock); Stop = LC.BudgetStop
            decide = lambda key, t: RC.decide(req, BD, key, B.context(t), ""); save, exists = RC.save_exec, RC.exec_exists
        jobs = [(t, s, e) for t in days for s in SEEDS for e in DRAWS]
        print(f"{a.bench} {a.run}: {len(days)} test dates x {len(SEEDS)} seeds x {len(DRAWS)} draws = {len(jobs)} memory-free decisions; "
              f"cap ${cap:.2f}; folder {out}", flush=True)
        def one(job):
            t, s, e = job; key = f"sc|{t}|{s}|{e}"
            if exists(out, key): return "skip"
            try: w, reason, n = decide(key, t)
            except Stop: return "budget"
            save(out, {"key": key, "phase": "test", "arm": "sc_extra", "date": t, "seed": s, "draw": e, "weights": w,
                       "reason": reason, "calls": n, "status": "VALID" if w else "FAILED"})
            return "VALID" if w else "FAILED"
        status, sec = run_pool(jobs, one, a.workers, f"{a.bench} {a.run}", lambda: ledger.spent)
        spent = ledger.spent
    summ = {"bench": a.bench, "run": a.run, "jobs": status, "spent_usd_conservative": round(spent, 6), "cap_usd": cap,
            "seconds": round(sec, 1), "stopped": stop.is_set()}
    (out / "RUN_SUMMARY.json").write_text(json.dumps(summ, indent=1)); print(json.dumps(summ, indent=1))


if __name__ == "__main__":
    main()
