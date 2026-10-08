#!/usr/bin/env python3
"""Self-consistency with its own three draws per seed (offline; no API calls).

Published Self-consistency averages the memory-free drafts of the three seeds on a date, so all seeds deploy the same
portfolio and its sd over seeds is 0. Here seed s averages three memory-free draws of its own: the logged draft (draw 0;
mask 0 on InvestorBench and ClassAlloc, view '0' of replicate s on PortBench) and the two extra draws of run_sc_extra.py
(draws 1-2; PortBench replicates 3+2s and 4+2s). Everything else is unchanged: the average of the raw draws is projected to
the mandate, held and scored with the benchmark's own functions (ib_turnover / cb_eval / monthly_lib, imported unchanged),
with holdings carried per seed on InvestorBench and ClassAlloc. A failed extra draw is left out of the average.
The published definition is recomputed by the same code as a check ("published" below must match the paper).

  python3 sc_eval.py --bench ib --run main|t0.0|t0.3|t1.0|nano|gpt5mini|llama70b|gemini25flash|qwen3|deepseekv31|haiku45 [--mock]
  python3 sc_eval.py --bench cb --run main|gpt-4.1-nano|gpt-5-mini|llama-3.3-70b|gemini-2.5-flash|qwen3-235b|deepseek-v3.1|claude-haiku-4.5 [--mock]
  python3 sc_eval.py --bench pb
Writes sc_runs/<bench>_<run>/SC_EVAL.json."""
from __future__ import annotations
import argparse, glob, json, pickle, sys
from pathlib import Path
import numpy as np
HERE = Path(__file__).resolve().parent; LAB = HERE.parent
SEEDS = [0, 1, 2]


def stats(A):
    """A: dates x investors x seeds, utility in pp -> per-seed means, mean and sd over seeds."""
    per = np.nanmean(A, (0, 1)); return dict(per_seed=per.tolist(), mean=float(per.mean()), sd=float(per.std(ddof=1)))


def extras(folder, vec):
    X = {}
    packed = folder / "executions.json"                  # compact form of the logged extra draws (shipped for the main runs)
    recs = json.load(open(packed)) if packed.exists() else [json.load(open(f)) for f in glob.glob(str(folder / "executions" / "*.json"))]
    for r in recs:
        if r.get("status") == "VALID" and r.get("weights"): X[(r["date"], r["seed"], r["draw"])] = vec(r["weights"])
    return X


def holdings_eval(C, X, project, util, drift, INV, scale, nxt=None):
    """InvestorBench / ClassAlloc: carried holdings (on ClassAlloc to the next decision date, nxt); returns (new, published)
    arrays of utility in pp."""
    W, seeds, dates = C["W"], C["seeds"], C["dates"]; R = C.get("ret", C.get("rel"))
    nD = len(dates); new = np.full((nD, 3, len(seeds)), np.nan); pub = new.copy(); ndraw = np.zeros((nD, len(seeds)))
    for pi, (p, (M, mfl, gam)) in enumerate(INV.items()):
        ew = project(np.full(len(next(iter(W.values()))["m0"]), 1.0 / len(next(iter(W.values()))["m0"])), M, mfl)
        for si, s in enumerate(seeds):
            h_new = ew.copy(); h_pub = ew.copy()
            for di, t in enumerate(dates):
                draws = [W[(t, s)]["m0"]] + [X[(t, s, e)] for e in (1, 2) if (t, s, e) in X]; ndraw[di, si] = len(draws)
                x_new = project(np.mean(draws, 0), M, mfl); x_pub = project(np.mean([W[(t, ss)]["m0"] for ss in seeds], 0), M, mfl)
                new[di, pi, si] = scale * float(np.ravel(util(x_new, R[t], gam, h_new)[0])[0]); h_new = drift(x_new, (nxt or R)[t])
                pub[di, pi, si] = scale * float(np.ravel(util(x_pub, R[t], gam, h_pub)[0])[0]); h_pub = drift(x_pub, (nxt or R)[t])
    return new, pub, ndraw


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--bench", required=True, choices=["ib", "cb", "pb"])
    ap.add_argument("--run", default="main"); ap.add_argument("--mock", action="store_true"); a = ap.parse_args()
    folder = HERE / "sc_runs" / (f"{a.bench}_{a.run}" + ("_mock" if a.mock else ""))
    out = {"bench": a.bench, "run": a.run, "mock": a.mock}
    if a.bench == "ib":
        sys.path.insert(0, str(LAB / "investorbench")); import ib_turnover as IT, ib_all as IA
        cache = (LAB / "investorbench" / "turnover" / "cache.pkl") if a.run == "main" else (HERE / "ib_runs" / a.run / "eval" / "cache.pkl")
        C = pickle.load(open(cache, "rb")); X = extras(folder, IA.vec)
        new, pub, nd = holdings_eval(C, X, IT.project, lambda x, r, g, h: IT.util(x, r, g, h), IT.drift, IT.INV, 1e4)   # bp per day
        out.update(unit="bp per day", dates=C["dates"])
        base = (LAB / "investorbench" / "turnover" / "base.json") if a.run == "main" else (HERE / "ib_runs" / a.run / "eval" / "base.json")
        if base.exists():
            B = json.load(open(base)); k = "Self-consistency"
            if k in B.get("res", {}): out["published_from_file"] = stats(1e4 * np.array(B["res"][k], float))
    elif a.bench == "cb":
        import cb_eval as CE
        name = "run" if a.run == "main" else f"run_{a.run}_t0.7"
        cand = [HERE / "classalloc" / name] + [Path(p) for p in glob.glob(str(HERE / "classalloc" / f"run_{a.run}_t*"))]
        run = next(p for p in cand if (p / "eval" / "cache.pkl").exists())
        C = pickle.load(open(run / "eval" / "cache.pkl", "rb")); X = extras(folder, lambda w: np.array([w[c] for c in CE.CL]))
        new, pub, nd = holdings_eval(C, X, CE.project, lambda x, r, g, h: CE.util(x, r, g, h), CE.drift, CE.INV, 100.0,
                                     nxt=CE.to_next(C["dates"]))  # pp per month
        post = np.array([t >= "2024-07-01" for t in C["dates"]])
        out.update(unit="pp per month", dates=C["dates"], run_folder=str(run.name),
                   before_cutoff=stats(new[~post]), after_cutoff=stats(new[post]),
                   published_before=stats(pub[~post]), published_after=stats(pub[post]))
        res = run / "eval" / "RESULTS.json"
        if res.exists():
            J = json.load(open(res)); out["published_from_file"] = J["all"]["rows"]["Self-consistency"]
    else:
        sys.path.insert(0, str(LAB / "portbench")); import monthly_lib as ML
        rows, ids, dates, split = ML.load_monthly()
        xr = ML.load_rows("sc")                          # extra memory-free drafts: replicates 3-8 of arm 0
        uni0 = {}
        for r in rows:
            if r.get("status") == "VALID_NATIVE_EXECUTION": uni0.setdefault(r["date"], set()).update(r["action"])
        outside = sum(1 for r in xr if r.get("status") == "VALID_NATIVE_EXECUTION" and not set(r["action"]) <= uni0.get(r["date"], set()))
        D = ML.Data(rows + xr, ids, dates, split, cache=ML.load_cache())
        res = {}
        for c in ("full-price", "raw-price"):
            for p in ("conservative", "balanced", "aggressive"):
                key = (c, p); P = json.load(open(LAB / "portbench" / f"pb_all_{c}_{p}.json"))
                test = [d for d in P["res"]["1/N"] if d >= split]; A = np.full((len(test), len(SEEDS)), np.nan); Bp = A.copy(); nd = A.copy()
                for di, d in enumerate(test):
                    Wd = D.proj[key][d]
                    for s in SEEDS:
                        vs = [Wd[r]["0"] for r in (s, 3 + 2 * s, 4 + 2 * s) if "0" in Wd.get(r, {})]; nd[di, s] = len(vs)
                        A[di, s] = 100 * float(D.U(np.mean(vs, 0), d, p)[0])
                        Bp[di, s] = 100 * float(D.U(np.mean([Wd[r]["0"] for r in sorted(Wd) if r in (0, 1, 2) and "0" in Wd[r]], 0), d, p)[0])
                pubf = np.array([[100 * P["res"]["Self-consistency (3 drafts)"][d][s] for s in SEEDS] for d in test]) if "Self-consistency (3 drafts)" in P["res"] else None
                res[f"{c}|{p}"] = dict(new=A, pub=Bp, pubf=pubf, nd=nd)
        out.update(unit="pp per month", extra_rows=len(xr), extra_rows_outside_universe=outside)
        for c, lab in (("full-price", "PortBench-Full"), ("raw-price", "PortBench-Raw")):
            ks = [k for k in res if k.startswith(c)]
            new = np.stack([res[k]["new"] for k in ks], 1); pub = np.stack([res[k]["pub"] for k in ks], 1)
            out[lab] = dict(new=stats(new), published=stats(pub), draws_per_seed=float(np.mean([res[k]["nd"] for k in ks])),
                            max_abs_diff_vs_pb_all=float(max(np.nanmax(np.abs(res[k]["pub"] - res[k]["pubf"])) for k in ks if res[k]["pubf"] is not None)))
        folder.mkdir(parents=True, exist_ok=True); (folder / "SC_EVAL.json").write_text(json.dumps(out, indent=1)); print(json.dumps(out, indent=1)); return
    out.update(new=stats(new), published=stats(pub), draws_per_seed=float(nd.mean()), extra_draws=len(X),
               res_new=new.tolist())
    folder.mkdir(parents=True, exist_ok=True); (folder / "SC_EVAL.json").write_text(json.dumps(out, indent=1))
    print(json.dumps({k: v for k, v in out.items() if k not in ("res_new", "dates")}, indent=1))


if __name__ == "__main__":
    main()
