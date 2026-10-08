#!/usr/bin/env python3
"""Memorization probe for ClassAlloc (paid; small): does an LLM recall the market outcomes of a month from its date alone?

For every ClassAlloc decision date (58 warm-up and 71 test months, Jul 2015 - Mar 2026), the LLM is given only the date and
asked for the return of each asset class over the 20 trading days that ClassAlloc holds a decision. No market data, no
memory. Three draws per date (seeds 0-2) at the temperature of the runs (gpt-5-mini: the temperature fixed by its API).
The answers are compared with the realized ClassAlloc returns before and after each LLM's knowledge cutoff:
  * absolute recall: time-series Pearson correlation of recalled and realized returns over the months of the period,
    averaged over the four risky classes;
  * relative recall: Pearson correlation of recalled and realized returns after removing, within the period, each class's
    mean and each month's mean over the five classes, i.e. of which class did better than usual in which month (what
    matters for beating 1/N); class means alone (equities usually beat cash) do not count.
Each statistic is computed per seed; the summary gives mean and sd over the seeds.

  python3 probe_memory.py --mock                     # no network: end-to-end test into probe_runs/<model>_mock
  python3 probe_memory.py [--models a,b,...]         # paid; default: all eight LLMs of the paper; cap USD 1 per model
  python3 probe_memory.py --eval                     # offline: probe_runs/PROBE_RESULTS.json from the saved answers
"""
from __future__ import annotations
import argparse, json, math, re, sys, threading, time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

HERE = Path(__file__).resolve().parent; sys.path.insert(0, str(HERE))
import llm_client as LC
import run_cb as RC

MODELS = ["gpt-4.1-mini", "gpt-4.1-nano", "gpt-5-mini", "llama-3.3-70b", "qwen3-235b", "deepseek-v3.1", "gemini-2.5-flash", "claude-haiku-4.5"]
CUTOFF = {"gpt-4.1-mini": "2024-06", "gpt-4.1-nano": "2024-06", "gpt-5-mini": "2024-05", "llama-3.3-70b": "2023-12",
          "gemini-2.5-flash": "2025-01", "claude-haiku-4.5": "2025-07", "qwen3-235b": None, "deepseek-v3.1": None}   # as collect_models.py
SEEDS = [0, 1, 2]
SYSTEM = "You answer questions about financial markets."
ASK = ("Date: {t}\n\nConsider five asset classes, each an equal-weighted index of exchange-traded assets: equities, bonds, "
       "commodities, real estate and cash. From what you know about the markets, give the return in % of each asset class over "
       "the 20 trading days starting on this date. If you are not sure, give your best estimate. Respond with a JSON object only, "
       'for example {{"equities": 1.5, "bonds": -0.3, "commodities": 2.0, "real_estate": 0.4, "cash": 0.1, "reason": "one short sentence"}}.')


def body(spec, T, t):
    return LC.adapt({"model": "", "temperature": 0.0, "max_tokens": 200,
                     "messages": [{"role": "system", "content": SYSTEM}, {"role": "user", "content": ASK.format(t=t)}]}, spec, T)


def parse(txt):
    m = re.search(r"\{.*\}", txt or "", re.S)
    if not m: return None
    try: obj = json.loads(m.group(0))
    except Exception: return None
    out = {}
    for c in RC.CLASSES:
        v = obj.get(c, obj.get(RC.LABEL[c]))
        try: v = float(str(v).replace("%", "").strip())
        except (TypeError, ValueError): return None
        if not math.isfinite(v): return None
        out[c] = v
    return out


def collect(model, mock, cap, workers):
    spec = LC.MODELS[model]; T = None if spec["reasoning"] else RC.BASE_T; B = RC.Bench()
    out = HERE / "probe_runs" / (model + ("_mock" if mock else "")); (out / "calls").mkdir(parents=True, exist_ok=True)
    ledger = LC.Ledger(cap, out); stop = threading.Event(); req = LC.make_requester(ledger, out, spec, stop, mock)
    dates = B.dates[:12] if mock else B.dates; jobs = [(t, s) for t in dates for s in SEEDS]; ans = {}; t0 = time.time()
    print(f"probe {model}: {len(dates)} dates x {len(SEEDS)} draws; cap ${cap:.2f}; folder {out}", flush=True)

    def one(job):
        t, s = job
        try: txt = LC.content(req(f"probe|{t}|{s}", body(spec, T, t)))
        except LC.BudgetStop: return job, "budget", None
        p = parse(txt); return job, ("VALID" if p else "INVALID"), (p, txt[:300])
    status = {}
    with ThreadPoolExecutor(workers) as ex:
        for f in as_completed([ex.submit(one, j) for j in jobs]):
            (t, s), st, res = f.result(); status[st] = status.get(st, 0) + 1
            if res: ans[f"{t}|{s}"] = {"date": t, "seed": s, "recall": res[0], "reply": res[1]}
    (out / "ANSWERS.json").write_text(json.dumps(ans, indent=1))
    summ = {"model": spec["model"], "temperature": T, "jobs": status, "spent_usd_conservative": round(ledger.spent, 6), "cap_usd": cap,
            "seconds": round(time.time() - t0, 1), "stopped": stop.is_set()}
    (out / "RUN_SUMMARY.json").write_text(json.dumps(summ, indent=1)); print(json.dumps(summ), flush=True)


def pearson(x, y):
    n = len(x)
    if n < 3: return float("nan")
    mx, my = sum(x) / n, sum(y) / n; sx = math.sqrt(sum((a - mx) ** 2 for a in x)); sy = math.sqrt(sum((b - my) ** 2 for b in y))
    return float("nan") if sx == 0 or sy == 0 else sum((a - mx) * (b - my) for a, b in zip(x, y)) / (sx * sy)


def stats(v):
    v = [x for x in v if math.isfinite(x)]
    if not v: return {"mean": None, "sd": None, "per_seed": []}
    m = sum(v) / len(v); sd = math.sqrt(sum((x - m) ** 2 for x in v) / (len(v) - 1)) if len(v) > 1 else 0.0
    return {"mean": m, "sd": sd, "per_seed": v}


def evaluate(mock):
    B = RC.Bench(); real = {t: B.outcome(t) for t in B.dates}; risky = RC.CLASSES[:4]; res = {}
    for model in MODELS:
        f = HERE / "probe_runs" / (model + ("_mock" if mock else "")) / "ANSWERS.json"
        if not f.exists(): continue
        A = [a for a in json.load(open(f)).values() if a.get("recall")]; cut = CUTOFF[model]
        periods = {"all": lambda t: True}
        if cut: periods.update({"before cutoff": lambda t, c=cut: t[:7] <= c, "after cutoff": lambda t, c=cut: t[:7] > c})
        for y in range(2015, 2027): periods[f"year {y}"] = lambda t, y=y: t[:4] == str(y)
        R = {"cutoff": cut, "valid_answers": len(A), "expected_answers": len(B.dates) * len(SEEDS)}
        for pname, inside in periods.items():
            rel, ab = [], []
            for s in SEEDS:
                rows = sorted([a for a in A if a["seed"] == s and inside(a["date"])], key=lambda a: a["date"])
                if len(rows) < 3: continue
                P = [[a["recall"][c] for c in RC.CLASSES] for a in rows]; Rr = [[real[a["date"]][c] for c in RC.CLASSES] for a in rows]
                ab.append(sum(pearson([p[j] for p in P], [r[j] for r in Rr]) for j in range(len(risky))) / len(risky))
                def dd(X):                                  # remove class means and month means (double demeaning)
                    n, k = len(X), len(X[0]); cm = [sum(x[j] for x in X) / n for j in range(k)]; g = sum(cm) / k
                    return [X[i][j] - cm[j] - sum(X[i]) / k + g for i in range(n) for j in range(k)]
                rel.append(pearson(dd(P), dd(Rr)))
            R[pname] = {"months": len({a["date"] for a in A if inside(a["date"])}), "relative_recall": stats(rel), "absolute_recall": stats(ab)}
        res[model] = R
    out = HERE / "probe_runs" / ("PROBE_RESULTS_mock.json" if mock else "PROBE_RESULTS.json")
    out.write_text(json.dumps(res, indent=1))
    for model, R in res.items():
        line = f"{model:18s} cutoff {R['cutoff'] or '--':8s} answers {R['valid_answers']}/{R['expected_answers']}"
        for p in ("before cutoff", "after cutoff", "all"):
            if p in R and R[p]["absolute_recall"]["mean"] is not None:
                q = R[p]; line += (f" | {p} ({q['months']} m): abs {q['absolute_recall']['mean']:+.2f}±{q['absolute_recall']['sd']:.2f}"
                                   f" rel {q['relative_recall']['mean']:+.2f}±{q['relative_recall']['sd']:.2f}")
        print(line)
    print("wrote", out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mock", action="store_true"); ap.add_argument("--eval", action="store_true")
    ap.add_argument("--models", default=",".join(MODELS)); ap.add_argument("--cap", type=float, default=1.0)
    ap.add_argument("--workers", type=int, default=8)
    a = ap.parse_args()
    if not a.eval:
        for m in a.models.split(","): collect(m, a.mock, a.cap, a.workers)
    evaluate(a.mock)


if __name__ == "__main__":
    main()
