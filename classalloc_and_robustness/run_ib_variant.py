#!/usr/bin/env python3
"""InvestorBench with another LLM or another temperature (supplementary runs; paid).

Everything is as in the main InvestorBench run (investorbench/run_ib.py, whose prompts and agent code are reused
unchanged): the four stocks and cash, the 149 test days, the 63 lessons written in the warm-up and their embeddings
(copied from investorbench/run, so retrieval and the experience pool are identical), the 16 subsets of the four
retrieved lessons, ExpeL, FinMem-, MemRL- and Reflexion-style chains, three seeds. Only the LLM that makes the test
decisions (and writes the ExpeL insights and Reflexion notes) or its temperature changes.

Variants (python3 run_ib_variant.py --list):
  t0.0, t0.3, t1.0  gpt-4.1-mini at temperature 0, 0.3, 1.0   (the main run is temperature 0.7)
  nano              gpt-4.1-nano at 0.7                         (cheaper model)
  gpt5mini          gpt-5-mini, reasoning effort "minimal"      (newer general model; temperature is fixed by the API)
  llama70b          Llama-3.3-70B-Instruct via OpenRouter, 0.7  (open-weight model)

Usage (from this folder):
  python3 run_ib_variant.py --variant t0.0 --mock     # no network: end-to-end test into ib_runs_mock/
  python3 run_ib_variant.py --variant t0.0 --probe    # one real call (< USD 0.01): key, model, parameters, cost per call
  python3 run_ib_variant.py --variant t0.0            # full run into ib_runs/t0.0 (resumable; hard cap per variant)
"""
from __future__ import annotations
import argparse, json, shutil, sys, threading, time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

HERE = Path(__file__).resolve().parent
IB = HERE.parent / "investorbench"
sys.path.insert(0, str(IB)); sys.path.insert(0, str(HERE))
import run_ib as RB                       # prompts, Bench, decide, chain, plan_jobs, run_job (unchanged)
import llm_client as LC
LC.BudgetStop = RB.BudgetStop             # one exception class, so run_ib's chains and jobs stop cleanly at the cap

VARIANTS = {
    "t0.0":     dict(model="gpt-4.1-mini", temperature=0.0, cap=6.5),
    "t0.3":     dict(model="gpt-4.1-mini", temperature=0.3, cap=6.5),
    "t1.0":     dict(model="gpt-4.1-mini", temperature=1.0, cap=6.5),
    "nano":     dict(model="gpt-4.1-nano", temperature=0.7, cap=2.5),
    "gpt5mini": dict(model="gpt-5-mini",   temperature=None, cap=10.0),
    "llama70b": dict(model="llama-3.3-70b", temperature=0.7, cap=5.0),
    # added 5 Oct 2026
    "gemini25flash": dict(model="gemini-2.5-flash", temperature=0.7, cap=8.0),
    "haiku45":       dict(model="claude-haiku-4.5", temperature=0.7, cap=18.0),
    "qwen3":         dict(model="qwen3-235b", temperature=0.7, cap=4.0),
    "deepseekv31":   dict(model="deepseek-v3.1", temperature=0.7, cap=5.0),
}


def patch_bodies(spec, temperature):
    """make run_ib build its usual prompts for this model / temperature (prompt text unchanged)."""
    dec, fb, ins = RB.decision_body, RB.feedback_body, RB.insight_body
    RB.decision_body = lambda ctx, block: LC.adapt(dec(ctx, block), spec, temperature)
    RB.feedback_body = lambda ctx, w, r, kind: LC.adapt(fb(ctx, w, r, kind), spec, temperature)
    RB.insight_body = lambda exps: LC.adapt(ins(exps), spec, temperature)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--variant"); ap.add_argument("--list", action="store_true")
    ap.add_argument("--mock", action="store_true"); ap.add_argument("--probe", action="store_true")
    ap.add_argument("--workers", type=int, default=8); ap.add_argument("--cap", type=float, default=None)
    a = ap.parse_args()
    if a.list or not a.variant:
        for k, v in VARIANTS.items(): print(f"{k:9s} {v['model']:14s} temperature {v['temperature']}  cap ${v['cap']}")
        return
    V = VARIANTS[a.variant]; spec = LC.MODELS[V["model"]]; cap = a.cap if a.cap is not None else V["cap"]
    patch_bodies(spec, V["temperature"])
    B = RB.Bench()
    if a.probe:
        out = HERE / "ib_runs_probe" / a.variant; (out / "calls").mkdir(parents=True, exist_ok=True)
        LC.list_models(spec["provider"])
        stop = threading.Event(); led = LC.Ledger(0.05, out); req = LC.make_requester(led, out, spec, stop, False)
        t = B.test[0]; t0 = time.time()
        w, reason, n = RB.decide(req, f"probe|{t}|{time.time():.0f}", B.context(t), "")
        print(f"probe {a.variant}: model {spec['model']}; weights {w}; reason {reason!r}; calls {n}; "
              f"{time.time() - t0:.1f}s; cost so far ${led.spent:.5f}")
        return
    src = RB.HERE / "run"
    out = (HERE / ("ib_runs_mock" if a.mock else "ib_runs")) / a.variant
    (out / "calls").mkdir(parents=True, exist_ok=True); (out / "executions").mkdir(exist_ok=True)
    for f in ("experiences.json", "embeddings.json"):            # identical experience pool and retrieval
        if not (out / f).exists(): shutil.copyfile(src / f, out / f)
        if LC.sha((out / f).read_bytes()) != LC.sha((src / f).read_bytes()): sys.exit(f"{f} differs from the main run")
    exps = json.loads((out / "experiences.json").read_text()); E = json.loads((out / "embeddings.json").read_text())
    test_days = B.test[:12] if a.mock else B.test; seeds = RB.SEEDS
    n_dec = len(test_days) * len(seeds) * 20; n_fb = len(test_days) * len(seeds)
    print(f"variant {a.variant}: {spec['model']} (temperature {V['temperature']}); {len(test_days)} test days, seeds {seeds}; "
          f"about {n_dec} decisions + {n_fb} Reflexion notes; cap ${cap:.2f}; folder {out}", flush=True)
    manifest = {"variant": a.variant, "model": spec["model"], "provider": spec["provider"], "temperature": V["temperature"],
                "reasoning_effort": spec.get("effort"), "cap_usd": cap, "seeds": seeds, "test_days": [test_days[0], test_days[-1], len(test_days)],
                "experience_pool_sha256": LC.sha((out / "experiences.json").read_bytes()),
                "run_ib_sha256": LC.sha((IB / "run_ib.py").read_bytes()), "started_utc": LC.now()}
    if not (out / "MANIFEST.json").exists(): (out / "MANIFEST.json").write_text(json.dumps(manifest, indent=1))
    ledger = LC.Ledger(cap, out); stop = threading.Event()
    req = LC.make_requester(ledger, out, spec, stop, a.mock)
    t0 = time.time()
    try:
        ins = RB.phase_insights(req, out, exps)
    except LC.BudgetStop:
        print("budget stop before the test phase"); return
    jobs = RB.plan_jobs(test_days, exps, E, ins, seeds); log = []
    threads = [threading.Thread(target=RB.chain, args=(B, req, out, k, s, test_days, exps, E, log), daemon=True)
               for k in ("finmem", "memrl", "reflexion") for s in seeds]
    for th in threads: th.start()
    status = {}
    with ThreadPoolExecutor(a.workers) as ex:
        futs = [ex.submit(RB.run_job, B, req, out, j) for j in jobs]
        for i, f in enumerate(as_completed(futs)):
            try: st = f.result()
            except Exception as e: st = "error"; log.append(f"job error: {type(e).__name__}: {str(e)[:200]}")
            status[st] = status.get(st, 0) + 1
            if (i + 1) % 500 == 0: print(f"  {i + 1}/{len(jobs)} jobs; spent ${ledger.spent:.3f}; {time.time() - t0:.0f}s", flush=True)
    for th in threads: th.join()
    n_exec = len(list((out / "executions").glob("*.json")))
    summ = {"variant": a.variant, "jobs": status, "executions_saved": n_exec, "spent_usd_conservative": round(ledger.spent, 6),
            "cap_usd": cap, "seconds": round(time.time() - t0, 1), "finished_utc": LC.now(), "log": log[:50], "stopped": stop.is_set()}
    (out / "RUN_SUMMARY.json").write_text(json.dumps(summ, indent=1)); print(json.dumps(summ, indent=1))


if __name__ == "__main__":
    main()
