#!/usr/bin/env python3
"""ClassAlloc with the calendar dates hidden from the LLM (paid): the look-ahead check of the paper.

Everything is as in run_cb.py, which is imported unchanged (data, prompts, protocol, seeds; cb_eval.py charges the same
15 bp fee), except that no calendar date reaches the LLM:
  * the market context drops its first line ("Date: YYYY-MM-DD"); the returns and volatilities shown are unchanged;
  * warm-up lessons are stored and retrieved without the "(YYYY-MM-DD) " prefix that run_cb.py adds;
  * the ExpeL insight prompt lists the lessons without their dates.
The lessons and Reflexion notes are written by the LLM from the dateless context. FinMem's recency still uses the month
index, which the LLM never sees. The run counts the lessons that nevertheless mention a year or a month name.

Usage (from this folder):
  python3 run_cb_masked.py --mock                    # no network: end-to-end test into classalloc/run_masked_mock
  python3 run_cb_masked.py --dry-run [--model M]     # counts, cost estimate and one prompt; no calls
  python3 run_cb_masked.py [--model M]               # full run into classalloc/run_masked-<M>_t0.7 (resumable; cap USD 6)
  python3 run_cb_masked.py --sc [--model M]          # Self-consistency: two more memory-free draws per month and seed,
                                                     #   into sc_runs/cb_masked-<M> (resumable; cap USD 1)
Default model: gpt-4.1-mini (the main runs). Offline evaluation afterwards (refresh_all.sh does both):
  python3 cb_eval.py --run run_masked-<M>_t0.7 ;  python3 sc_eval.py --bench cb --run masked-<M>
"""
from __future__ import annotations
import argparse, json, re, sys, threading, time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

HERE = Path(__file__).resolve().parent; sys.path.insert(0, str(HERE))
import llm_client as LC
import run_cb as RC

MONTHS = "January|February|March|April|May|June|July|August|September|October|November|December"
DATE_WORDS = re.compile(r"\b(?:19|20)\d\d\b|\b(?:" + MONTHS + r")\b")


class MaskedBench(RC.Bench):
    def context(self, t):
        s = super().context(t); head = f"Date: {t}\n\n"
        assert s.startswith(head), "run_cb.Bench.context has changed"
        return s[len(head):]


class MaskedBodies(RC.Bodies):
    def insights(self, exps):                            # run_cb.Bodies.insights without the dates
        lines = [f"- [portfolio {e['port']:+.2f}% vs equal weight {e['ew']:+.2f}%] {e['lesson']}" for e in exps]
        user = ("Below are lessons you wrote after past monthly decisions, each with the outcome of that decision.\n\n" + "\n".join(lines) +
                "\n\nCompare the successful and unsuccessful decisions and extract at most 8 general insights for future monthly "
                "allocation across equities, bonds, commodities, real estate and cash. Return a numbered list; each insight at most 30 words.")
        return self._b(user, 600)


def warmup(B, BD, req, out, warm, workers):
    exps = RC.phase_warmup(B, BD, req, out, warm, workers)
    for e in exps: e["text"] = e["lesson"]               # no "(date) " prefix
    (out / "experiences.json").write_text(json.dumps(exps, indent=1))
    return exps, sum(1 for e in exps if DATE_WORDS.search(e["lesson"]))


def run_sc(a, spec, BD, B, name, test):
    """as run_sc_extra.py --bench cb (same keys, draws and record format), with the dateless context."""
    import run_sc_extra as RS
    out = HERE / "sc_runs" / (f"cb_{name}" + ("_mock" if a.mock else "")); stop = threading.Event()
    (out / "calls").mkdir(parents=True, exist_ok=True); (out / "executions").mkdir(exist_ok=True)
    cap = a.cap if a.cap is not None else 1.0; ledger = LC.Ledger(cap, out); req = LC.make_requester(ledger, out, spec, stop, a.mock)
    jobs = [(t, s, e) for t in test for s in RS.SEEDS for e in RS.DRAWS]
    print(f"cb {name}: {len(test)} test dates x {len(RS.SEEDS)} seeds x {len(RS.DRAWS)} draws = {len(jobs)} memory-free decisions "
          f"(dates hidden); cap ${cap:.2f}; folder {out}", flush=True)

    def one(job):
        t, s, e = job; key = f"sc|{t}|{s}|{e}"
        if RC.exec_exists(out, key): return "skip"
        try: w, reason, n = RC.decide(req, BD, key, B.context(t), "")
        except LC.BudgetStop: return "budget"
        RC.save_exec(out, {"key": key, "phase": "test", "arm": "sc_extra", "date": t, "seed": s, "draw": e, "weights": w,
                           "reason": reason, "calls": n, "status": "VALID" if w else "FAILED"})
        return "VALID" if w else "FAILED"
    status, sec = RS.run_pool(jobs, one, a.workers, f"cb {name}", lambda: ledger.spent)
    summ = {"bench": "cb", "run": name, "dates_hidden": True, "jobs": status, "spent_usd_conservative": round(ledger.spent, 6),
            "cap_usd": cap, "seconds": round(sec, 1), "stopped": stop.is_set()}
    (out / "RUN_SUMMARY.json").write_text(json.dumps(summ, indent=1)); print(json.dumps(summ, indent=1))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mock", action="store_true"); ap.add_argument("--dry-run", action="store_true"); ap.add_argument("--sc", action="store_true")
    ap.add_argument("--model", default=RC.BASE_MODEL); ap.add_argument("--temperature", type=float, default=RC.BASE_T)
    ap.add_argument("--cap", type=float, default=None); ap.add_argument("--workers", type=int, default=8)
    a = ap.parse_args()
    spec = LC.MODELS[a.model]; T = None if spec["reasoning"] else a.temperature; BD = MaskedBodies(spec, T); B = MaskedBench()
    name = f"masked-{a.model}"
    warm, test, seeds = B.warm, (B.test[:12] if a.mock else B.test), RC.SEEDS
    if a.sc: return run_sc(a, spec, BD, B, name, test)
    out = HERE / "classalloc" / ("run_masked_mock" if a.mock else f"run_{name}_t{a.temperature}")
    cap = a.cap if a.cap is not None else 6.0
    n_dec = len(warm) + len(test) * len(seeds) * (16 + 1 + 3); n_fb = len(warm) + len(test) * len(seeds)
    p_dec = len(LC.canon(BD.decision(B.context(test[0]), RC.block_experiences(["x" * 420] * 4)))) / 3.6
    est = n_dec * (p_dec * spec["p_in"] + 80 * spec["p_out"]) + n_fb * (900 * spec["p_in"] + 90 * spec["p_out"])
    print(f"ClassAlloc, dates hidden: model {spec['model']} (temperature {T}); warm-up {len(warm)} months, test {len(test)} months "
          f"({test[0]}..{test[-1]}), seeds {seeds}; decisions {n_dec}, lessons/reflections {n_fb}; estimated cost ${est:.2f}; "
          f"cap ${cap:.2f}; folder {out}", flush=True)
    if a.dry_run:
        print("\n--- example prompt (test month 1, no memory) ---\n" + RC.SYSTEM + "\n\n" + B.context(test[0]) + RC.ASK); return
    (out / "calls").mkdir(parents=True, exist_ok=True); (out / "executions").mkdir(exist_ok=True)
    man = {"benchmark": "ClassAlloc", "dates_hidden": True, "model": spec["model"], "temperature": T, "cap_usd": cap, "seeds": seeds,
           "warmup": [warm[0], warm[-1], len(warm)], "test": [test[0], test[-1], len(test)],
           "data_sha256": LC.sha(RC.DATA.read_bytes()), "run_cb_sha256": LC.sha(Path(RC.__file__).read_bytes()),
           "run_cb_masked_sha256": LC.sha(Path(__file__).read_bytes()), "started_utc": LC.now(),
           "feedback": "each outcome is learned once its holding window has ended"}
    if not (out / "MANIFEST.json").exists(): (out / "MANIFEST.json").write_text(json.dumps(man, indent=1))
    ledger = LC.Ledger(cap, out); stop = threading.Event(); req = LC.make_requester(ledger, out, spec, stop, a.mock); t0 = time.time()
    try:
        exps, n_dated = warmup(B, BD, req, out, warm, a.workers)
        print(f"warm-up: {len(exps)} lessons ({n_dated} mention a year or a month name); spent ${ledger.spent:.3f} ({time.time() - t0:.0f}s)", flush=True)
        E = RC.phase_embed(B, req, out, exps, test); ins = RC.phase_insights(BD, req, out, exps)
        print(f"embeddings {len(E['experiences'])} + {len(E['contexts'])}; insights {len(ins)}; spent ${ledger.spent:.3f}", flush=True)
    except LC.BudgetStop:
        print("budget stop during setup"); return
    jobs = RC.plan_jobs(test, exps, E, ins, seeds); log = []
    threads = [threading.Thread(target=RC.chain, args=(B, BD, req, out, k, s, test, exps, E, log), daemon=True)
               for k in ("finmem", "memrl", "reflexion") for s in seeds]
    for th in threads: th.start()
    status = {}
    with ThreadPoolExecutor(a.workers) as ex:
        futs = [ex.submit(RC.run_job, B, BD, req, out, j) for j in jobs]
        for i, f in enumerate(as_completed(futs)):
            try: st = f.result()
            except Exception as e: st = "error"; log.append(f"job error: {type(e).__name__}: {str(e)[:200]}")
            status[st] = status.get(st, 0) + 1
            if (i + 1) % 500 == 0: print(f"  {i + 1}/{len(jobs)} jobs; spent ${ledger.spent:.3f}; {time.time() - t0:.0f}s", flush=True)
    for th in threads: th.join()
    summ = {"dates_hidden": True, "lessons_mentioning_a_date": n_dated, "jobs": status,
            "executions_saved": len(list((out / "executions").glob("*.json"))), "spent_usd_conservative": round(ledger.spent, 6),
            "cap_usd": cap, "seconds": round(time.time() - t0, 1), "finished_utc": LC.now(), "log": log[:50], "stopped": stop.is_set()}
    (out / "RUN_SUMMARY.json").write_text(json.dumps(summ, indent=1)); print(json.dumps(summ, indent=1))


if __name__ == "__main__":
    main()
