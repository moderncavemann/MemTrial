#!/usr/bin/env python3
"""Offline evaluation of an InvestorBench variant run (no API calls).

Same methods, mandates, fee rule (15 bp on what is traded) and frozen method code (memtrial.py) as the InvestorBench
column of Table 3: the functions of investorbench/ib_turnover.py are reused unchanged, only pointed at the
variant's logged drafts. Writes <run>/eval/{cache.pkl, base.json, mt_*.json, SUMMARY.json}; never touches the published
files of the main run.

Usage: python3 ib_eval_variant.py VARIANT [--mock]      (VARIANT = folder name under ib_runs/ or ib_runs_mock/)
       python3 ib_eval_variant.py --run-dir DIR --out-dir DIR2   (any run folder, e.g. the main run, for a reproduction check)
       add --stage cache|base|mt|mt-nogate|summary to run one step at a time (default: all steps)
"""
from __future__ import annotations
import collections, glob, json, math, pickle, sys
from pathlib import Path
import numpy as np

HERE = Path(__file__).resolve().parent
IB = HERE.parent / "investorbench"
sys.path.insert(0, str(IB)); sys.path.insert(0, str(IB.parent / "memtrial"))
import ib_turnover as IT                  # unchanged; IT.OUT is redirected below
import memtrial as MT

EXPERIENCE = ["FinMem", "MemRL", "Reflexion", "ExpeL"]
METHODS = ["1/N", "Minimum variance", "Zero-shot (no memory)", "Self-consistency", "Similarity retrieval (top-2)",
           "Similarity retrieval (top-4)", "FinMem", "MemRL", "Reflexion", "ExpeL", "Uplift credit",
           "Counterfactual selection", "Draft averaging", "Hedge", "MemTrial", "MemTrial | no gate"]


def build_cache(run):
    """as ib_turnover.stage_cache, for the logged drafts in `run`."""
    import run_ib as RB, ib_all as IA
    B = RB.Bench()
    recs = [json.load(open(f)) for f in glob.glob(str(run / "executions" / "*.json"))]
    exps = json.load(open(run / "experiences.json")); E = json.load(open(run / "embeddings.json"))
    ids_all = [e["id"] for e in exps]; Xe = np.array([E["experiences"][h] for h in ids_all]); Xe = Xe - Xe.mean(0)
    _, _, Vt = np.linalg.svd(Xe, full_matrices=False); Zc = Xe @ Vt[:8].T; Zc = Zc / (Zc.std(0) + 1e-12)
    Z = {h: list(z) for h, z in zip(ids_all, Zc)}
    W = collections.defaultdict(dict); top4 = {}
    for r in recs:
        if r.get("phase") != "test" or r.get("status") != "VALID": continue
        arm = f"m{r['mask']}" if r["arm"] == "subset" else r["arm"]
        W[(r["date"], r["seed"])][arm] = IA.vec(r["weights"])
        if r["arm"] == "subset": top4[r["date"]] = r["top4"]
    seeds = sorted({s for _, s in W})
    dates = [t for t in B.test if all((t, s) in W and all(f"m{m}" in W[(t, s)] for m in range(16)) for s in seeds)]
    ret = {t: np.array([B.next_returns(t)[a] / 100.0 for a in RB.ASSETS]) for t in dates}
    past = {t: np.array([[B.price[a][B.days[i]] / B.price[a][B.days[i - 1]] - 1 for a in RB.ASSETS]
                         for i in range(B.idx[t] - 59, B.idx[t] + 1)]) for t in dates}
    miss = collections.Counter(k for t in dates for s in seeds for k in ("finmem", "memrl", "reflexion", "expel") if k not in W[(t, s)])
    n_valid = sum(1 for r in recs if r.get("phase") == "test" and r.get("status") == "VALID")
    n_fail = sum(1 for r in recs if r.get("phase") == "test" and r.get("status") != "VALID")
    pickle.dump(dict(W=dict(W), top4=top4, seeds=seeds, dates=dates, ret=ret, past=past, Z=Z), open(IT.OUT / "cache.pkl", "wb"))
    print(f"{len(dates)} test days complete for all seeds {seeds}; valid drafts {n_valid}, invalid {n_fail}; missing baseline arms {dict(miss)}")
    return dict(valid=n_valid, invalid=n_fail, missing=dict(miss), days=len(dates))


def paired_p(a, b):
    """one-sided paired t-test of mean(a - b) > 0; a, b: per-date values."""
    d = np.asarray(a) - np.asarray(b); n = len(d); sd = d.std(ddof=1)
    if sd == 0: return 0.0 if d.mean() > 0 else 1.0
    return MT.t_sf(d.mean() / (sd / math.sqrt(n)), n - 1)


def diag(C):
    """Table 1 quantities for this run (common 1/N book, as ib_turnover.stage_diag)."""
    W, seeds, dates, ret = C["W"], C["seeds"], C["dates"], C["ret"]; out = {}
    def spearman(a, b):
        ra = np.argsort(np.argsort(a)); rb = np.argsort(np.argsort(b)); return float(np.corrcoef(ra, rb)[0, 1])
    for p, (M, mfl, gam) in IT.INV.items():
        ew = IT.project(np.full(5, 0.2), M, mfl); h = ew.copy(); oc, cf, mk, det = [], [], [], 0
        for t in dates:
            r = ret[t]
            U = np.array([[float(IT.util(IT.project(W[(t, s)][f"m{m}"], M, mfl), r, gam, h)[0]) for m in IT.FULL] for s in seeds])
            u1n = float(IT.util(ew, r, gam, h)[0]); Um = U.mean(0)
            for j in range(4):
                on = [m for m in IT.FULL if (m >> j) & 1]; off = [m for m in IT.FULL if not (m >> j) & 1]
                oc.append(Um[on].mean()); cf.append(Um[on].mean() - Um[off].mean()); mk.append(u1n)
            if len(seeds) > 1:
                grand = U.mean(); ssb = len(seeds) * ((Um - grand) ** 2).sum(); ssw = ((U - Um) ** 2).sum()
                dfw = 16 * (len(seeds) - 1); F = (ssb / 15) / (ssw / dfw) if ssw > 0 else np.inf; det += MT.f_sf(F, 15, dfw) < 0.05
            h = IT.drift(ew, r)
        out[p] = dict(outcome_vs_1N=spearman(oc, mk), cf_vs_1N=spearman(cf, mk), detectable_days=int(det), days=len(dates))
    return out


def main():
    if len(sys.argv) < 2: sys.exit(__doc__)
    if "--run-dir" in sys.argv:
        run = Path(sys.argv[sys.argv.index("--run-dir") + 1]).resolve(); IT.OUT = Path(sys.argv[sys.argv.index("--out-dir") + 1]).resolve()
    else:
        run = HERE / ("ib_runs_mock" if "--mock" in sys.argv else "ib_runs") / sys.argv[1]; IT.OUT = run / "eval"
    IT.OUT.mkdir(parents=True, exist_ok=True)
    stage = sys.argv[sys.argv.index("--stage") + 1] if "--stage" in sys.argv else "all"   # cache | base | mt | mt-nogate | summary
    if stage in ("all", "cache"):
        info = build_cache(run); (IT.OUT / "cache_info.json").write_text(json.dumps(info))
    if stage == "cache": return
    C = IT.load(); info = json.load(open(IT.OUT / "cache_info.json"))
    if stage in ("all", "base"): IT.stage_base()
    for name, st in (("MemTrial", "mt"), ("MemTrial | no gate", "mt-nogate")):
        if stage in ("all", st):
            res, op = IT.run_variant(name, C); print(f"{name:20s} {1e4 * res.mean():6.2f} bp/day; trust rate {100 * op.mean():.1f}%")
    if stage not in ("all", "summary"): return
    base = json.load(open(IT.OUT / "base.json"))
    R = {k: np.array(v, float) for k, v in base["res"].items()}; cost = {k: np.array(v, float) for k, v in base["cost"].items()}
    trust = {}; fwd = {}
    for name in ("MemTrial", "MemTrial | no gate"):
        J = json.load(open(IT.OUT / f"mt_{IT.slug(name)}.json")); R[name] = np.array(J["res"], float)
        cost[name] = np.array(J["cost"], float); trust[name] = 100 * float(np.mean(J["open"]))
        sc = [v[m][0] for v in J["forward_score"].values() for m in v if v[m][0] is not None]
        fwd[name] = float(np.mean(sc)) if sc else None
    rows = {}
    for k in METHODS:
        per_seed = 1e4 * np.nanmean(R[k], (0, 1))                         # bp/day, mean over days and investors
        rows[k] = dict(mean=float(per_seed.mean()), sd=float(per_seed.std(ddof=1)) if len(per_seed) > 1 else 0.0,
                       cost_bp=float(1e4 * np.nanmean(cost[k])))
    by_date = {k: np.nanmean(R[k], (1, 2)) for k in METHODS}              # per test day, mean over investors and seeds
    best = max(EXPERIENCE, key=lambda k: rows[k]["mean"])
    summ = {"run": str(run.name), "manifest": json.load(open(run / "MANIFEST.json")) if (run / "MANIFEST.json").exists() else None,
            "drafts": info, "rows_bp_per_day": rows, "best_experience_agent": best,
            "improv_vs_best_bp": rows["MemTrial"]["mean"] - rows[best]["mean"],
            "p_vs_best": paired_p(by_date["MemTrial"], by_date[best]),
            "memtrial_minus_1N_bp": rows["MemTrial"]["mean"] - rows["1/N"]["mean"],
            "best_minus_1N_bp": rows[best]["mean"] - rows["1/N"]["mean"],
            "trust_rate_pct": trust, "mean_forward_score": fwd, "table1": diag(C)}
    (IT.OUT / "SUMMARY.json").write_text(json.dumps(summ, indent=1))
    print(f"\n{run.name}: utility, bp per day (mean ± sd over seeds)")
    for k in METHODS: print(f"  {k:30s} {rows[k]['mean']:7.2f} ± {rows[k]['sd']:.2f}   cost {rows[k]['cost_bp']:.2f}")
    print(f"  best experience-learning agent: {best}; MemTrial - best = {summ['improv_vs_best_bp']:+.2f} bp (p = {summ['p_vs_best']:.3f}); "
          f"MemTrial - 1/N = {summ['memtrial_minus_1N_bp']:+.2f} bp; trust rate {trust['MemTrial']:.1f}%")


if __name__ == "__main__":
    main()
