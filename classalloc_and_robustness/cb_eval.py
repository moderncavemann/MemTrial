#!/usr/bin/env python3
"""Offline evaluation of ClassAlloc (no API calls): the methods of Table 3 on the logged drafts of run_cb.py.

Rules as on PortBench and InvestorBench: each portfolio is projected to the investor's mandate (PortBench's rule: risky
classes = equities and real estate, at most M; defensive classes = bonds and cash, at least m), held for 20 sessions,
utility = net return - (gamma / 2) * (variance of daily returns x 20), in pp per month. As on InvestorBench, holdings are
carried from month to month (drifted with prices) and the 15 bp fee is charged on what is traded; every method starts
the test period holding the projected 1/N portfolio. The method itself is the frozen memtrial.py (unmodified); the
baselines follow ib_turnover.stage_base line by line.

Usage: python3 cb_eval.py [--mock] [--run FOLDER] [--stage cache|base|mt0|mt1|mt2|summary]
       (FOLDER under classalloc/, default run or run_mock; without --stage all steps run)
Writes classalloc/<run>/eval/{cache.pkl, RESULTS.json} and prints the table.
"""
from __future__ import annotations
import collections, glob, json, math, pickle, random, sys, time
from pathlib import Path
import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE)); sys.path.insert(0, str(HERE.parent / "memtrial"))
import memtrial as MT
from memtrial import MemTrialBank, banzhaf, GRID, ftrl_q, f_sf
import run_cb as RC

CL = RC.CLASSES; K = len(CL)
RISKY = np.array([1.0 if c in ("equities", "real_estate") else 0.0 for c in CL])
DEF = np.array([1.0 if c in ("bonds", "cash") else 0.0 for c in CL])
INV = {"conservative": (0.40, 0.40, 10.0), "balanced": (0.65, 0.20, 5.0), "aggressive": (0.90, 0.05, 2.0)}
FEE = 0.0015; HALF = [0, 3, 5, 6, 9, 10, 12, 15]; FULL = list(range(16))
EXPERIENCE = ["FinMem", "MemRL", "Reflexion", "ExpeL"]
BASE = ["1/N", "Minimum variance", "Zero-shot (no memory)", "Self-consistency", "Similarity retrieval (top-2)",
        "Similarity retrieval (top-4)", "FinMem", "MemRL", "Reflexion", "ExpeL", "Draft averaging",
        "Counterfactual selection", "Uplift credit", "Hedge"]
MTV = {"MemTrial": dict(learner="auto"), "MemTrial | no gate": dict(learner="auto", gate="none"),
       "MemTrial | F-test gate": dict(learner="auto", gate="ftest")}


def _simplex(y):
    u = np.sort(y)[::-1]; css = np.cumsum(u)
    rho = np.nonzero(u * np.arange(1, len(y) + 1) > (css - 1))[0][-1]
    return np.maximum(y - (css[rho] - 1) / (rho + 1.0), 0)


def project(w, M, m, iters=3000, tol=1e-12):
    """Euclidean projection onto the mandate (as rescore.project, at the class level; Dykstra over three convex sets)."""
    y = np.maximum(np.asarray(w, float), 0); y = y / y.sum()
    if DEF @ y >= m - 1e-12 and RISKY @ y <= M + 1e-12: return y
    P = [lambda z: _simplex(z),
         lambda z: z if DEF @ z >= m else z + (m - DEF @ z) / DEF.sum() * DEF,
         lambda z: z if RISKY @ z <= M else z - (RISKY @ z - M) / RISKY.sum() * RISKY]
    x = y.copy(); p = [np.zeros_like(y) for _ in range(3)]
    for _ in range(iters):
        xo = x.copy()
        for k, Pk in enumerate(P): z = Pk(x + p[k]); p[k] = x + p[k] - z; x = z
        if np.abs(x - xo).max() < tol: break
    return _simplex(x)


def util(X, rel, gam, h):
    """utility of target(s) X (n x K) held 20 sessions with price relatives rel (K x 20), starting from holdings h."""
    X = np.atleast_2d(X); cost = FEE * np.abs(X - h).sum(1); nav = 1.0 - cost
    val = nav[:, None] * (X @ rel); path = np.concatenate([np.ones((len(X), 1)), val], 1)
    r = path[:, 1:] / path[:, :-1] - 1.0
    return val[:, -1] - 1.0 - 0.5 * gam * r.var(1) * rel.shape[1], cost, X @ rel[:, -1] - 1.0


def drift(x, rel): g = np.asarray(x, float) * rel[:, -1]; return g / g.sum()


# Calendar. A decision is scored over the 20 sessions after it, and its outcome becomes known when that window ends.
# Months with 19 sessions start the next decision one session earlier, so every learner receives an outcome only
# once it is complete (Feedback), and holdings are carried to the next decision date (to_next).
_BENCH = RC.Bench()


def known(t_done, t_now): return _BENCH.sidx[t_done] + _BENCH.hold <= _BENCH.sidx[t_now]


class Feedback:
    """outcomes in date order, each released once its holding window has ended."""
    def __init__(self): self.q = []
    def add(self, t, item): self.q.append((t, item))
    def ready(self, t_now):
        out = []
        while self.q and known(self.q[0][0], t_now): out.append(self.q.pop(0)[1])
        return out


def to_next(dates):
    """price relatives of every class from each decision date to the next (the last date: to the end of its window)."""
    out = {}
    for a, b in zip(dates, list(dates[1:]) + [None]):
        i = _BENCH.sidx[a]; j = _BENCH.sidx[b] if b is not None else i + _BENCH.hold
        out[a] = np.array([[_BENCH.L[c][j] / _BENCH.L[c][i]] for c in CL])
    return out


def minvar(past, M, m):
    C = np.cov(past.T); x = np.full(K, 1.0 / K); Lc = 2 * np.linalg.eigvalsh(C).max() + 1e-12
    for _ in range(300): x = x - (2 * C @ x) / Lc; x = _simplex(x)
    return project(x, M, m)


def build_cache(run):
    if not (run / "executions").exists(): sys.exit(f"{run.name}/eval/cache.pkl is built from the logs in {run.name}/executions, which are not included")
    B = RC.Bench(); recs = [json.load(open(f)) for f in glob.glob(str(run / "executions" / "*.json"))]
    exps = json.load(open(run / "experiences.json")); E = json.load(open(run / "embeddings.json"))
    ids = [e["id"] for e in exps]; Xe = np.array([E["experiences"][h] for h in ids]); Xe = Xe - Xe.mean(0)
    _, _, Vt = np.linalg.svd(Xe, full_matrices=False); Zc = Xe @ Vt[:8].T; Zc = Zc / (Zc.std(0) + 1e-12)
    Z = {h: list(z) for h, z in zip(ids, Zc)}
    W = collections.defaultdict(dict); top4 = {}
    for r in recs:
        if r.get("phase") != "test" or r.get("status") != "VALID": continue
        arm = f"m{r['mask']}" if r["arm"] == "subset" else r["arm"]
        W[(r["date"], r["seed"])][arm] = np.array([r["weights"][c] for c in CL])
        if r["arm"] == "subset": top4[r["date"]] = r["top4"]
    seeds = sorted({s for _, s in W})
    dates = [t for t in B.test if all((t, s) in W and all(f"m{m}" in W[(t, s)] for m in range(16)) for s in seeds)]
    rel, past = {}, {}
    for t in dates:
        i = B.sidx[t]
        rel[t] = np.array([[B.L[c][i + k] / B.L[c][i] for k in range(1, B.hold + 1)] for c in CL])
        past[t] = np.array([[B.L[c][j] / B.L[c][j - 1] - 1 for c in CL] for j in range(i - 59, i + 1)])
    n_valid = sum(1 for r in recs if r.get("phase") == "test" and r.get("status") == "VALID")
    n_fail = sum(1 for r in recs if r.get("phase") == "test" and r.get("status") != "VALID")
    C = dict(W=dict(W), top4=top4, seeds=seeds, dates=dates, rel=rel, past=past, Z=Z, valid=n_valid, invalid=n_fail)
    pickle.dump(C, open(run / "eval" / "cache.pkl", "wb"))
    print(f"{len(dates)} test months complete for seeds {seeds}; valid drafts {n_valid}, invalid {n_fail}")
    return C


def evaluate_base(C):
    W, top4, seeds, dates, rel, past = C["W"], C["top4"], C["seeds"], C["dates"], C["rel"], C["past"]
    nD, nS = len(dates), len(seeds)
    res = {k: np.full((nD, 3, nS), np.nan) for k in BASE}; cost = {k: np.full((nD, 3, nS), np.nan) for k in BASE}
    held = collections.Counter(); ARM = {"FinMem": "finmem", "MemRL": "memrl", "Reflexion": "reflexion", "ExpeL": "expel"}; nxt = to_next(dates)
    for pi, (p, (M, mfl, gam)) in enumerate(INV.items()):
        ew = project(np.full(K, 1.0 / K), M, mfl)
        mv = {t: minvar(past[t], M, mfl) for t in dates}
        sc = {t: project(np.mean([W[(t, ss)]["m0"] for ss in seeds], 0), M, mfl) for t in dates}
        for si, s in enumerate(seeds):
            H = {k: ew.copy() for k in BASE}; hist = []; up_obs = []; hed = []; rnd = random.Random(f"{p}|{s}"); fb = Feedback()
            for di, t in enumerate(dates):
                for kind, obj in fb.ready(t): {"hist": hist, "up": up_obs, "hed": hed}[kind].append(obj)
                w = {k: project(v, M, mfl) for k, v in W[(t, s)].items()}; R = rel[t]; ids = top4[t]
                ens8 = np.mean([w[f"m{m}"] for m in HALF], 0)
                target = {"1/N": ew, "Minimum variance": mv[t], "Zero-shot (no memory)": w["m0"], "Self-consistency": sc[t],
                          "Similarity retrieval (top-2)": w["m3"], "Similarity retrieval (top-4)": w["m15"], "Draft averaging": ens8}
                for k, a in ARM.items():
                    if a in w: target[k] = w[a]
                    else: target[k] = H[k]; held[k] += 1
                hC = H["Counterfactual selection"]; u16 = {m: float(util(w[f"m{m}"], R, gam, hC)[0][0]) for m in FULL}
                if hist:
                    cf = collections.defaultdict(list)
                    for hh in hist:
                        for j, hid in enumerate(hh["ids"]): cf[hid].append(hh["cf"][j])
                    scs = [np.mean(cf[h]) if h in cf else 0.0 for h in ids]; od = sorted(range(4), key=lambda j: (-scs[j], j))
                    target["Counterfactual selection"] = w[f"m{(1 << od[0]) | (1 << od[1])}"]
                else:
                    target["Counterfactual selection"] = w["m3"]
                fb.add(t, ("hist", {"ids": ids, "cf": [banzhaf(u16, j, FULL) for j in range(4)]}))
                if len(up_obs) >= 3:
                    allid = sorted({h for q in up_obs for h in q[0]} | set(ids)); ix = {h: i for i, h in enumerate(allid)}
                    Xr = np.array([[1.0, q[2]] + [1.0 if h in q[0] else 0.0 for h in allid] for q in up_obs]); yr = np.array([q[1] for q in up_obs])
                    pen = np.eye(Xr.shape[1]); pen[0, 0] = pen[1, 1] = 0.0
                    coef = np.linalg.solve(Xr.T @ Xr + 1e-4 * pen * len(yr), Xr.T @ yr); upl = [coef[2 + ix[h]] for h in ids]
                else:
                    upl = [0.0] * 4
                od = rnd.sample(range(4), 2) if rnd.random() < 0.1 else sorted(range(4), key=lambda j: (-upl[j], j))[:2]
                target["Uplift credit"] = w[f"m{(1 << od[0]) | (1 << od[1])}"]
                Wf = np.array([ew, w["m0"], w["m15"], w.get("finmem", w["m0"]), w.get("memrl", w["m0"])])
                uf = util(Wf, R, gam, H["Hedge"])[0]
                if hed:
                    Uh = np.array(hed); sg = float(np.mean(Uh.std(1))) + 1e-12; lam = 4.0 * sg / math.sqrt(len(Uh)); zz = Uh.mean(0) / lam
                    qh = np.exp(zz - zz.max()); qh /= qh.sum()
                else:
                    qh = np.full(5, 0.2)
                target["Hedge"] = qh @ Wf; fb.add(t, ("hed", list(uf)))
                for k in BASE:
                    X = target[k]; u, c, _ = util(X, R, gam, H[k])
                    res[k][di, pi, si] = float(u[0]); cost[k][di, pi, si] = float(c[0]); H[k] = drift(X, nxt[t])
                fb.add(t, ("up", ({ids[od[0]], ids[od[1]]}, res["Uplift credit"][di, pi, si], res["1/N"][di, pi, si])))
    return res, cost, dict(held)


def evaluate_method(C, name):
    cfg = dict(MTV[name]); W, top4, seeds, dates, rel, Z = C["W"], C["top4"], C["seeds"], C["dates"], C["rel"], C["Z"]
    nD, nS = len(dates), len(seeds)
    res = np.full((nD, 3, nS), np.nan); cost = np.full((nD, 3, nS), np.nan); opened = np.zeros((nD, 3, nS)); fscore = {}; nxt = to_next(dates)
    for pi, (p, (M, mfl, gam)) in enumerate(INV.items()):
        ew = project(np.full(K, 1.0 / K), M, mfl)
        for si, s in enumerate(seeds):
            bank = MemTrialBank(Z, {name: cfg}, masks=HALF); h = ew.copy(); fb = Feedback()
            for di, t in enumerate(dates):
                for a_ in fb.ready(t): bank.matured(*a_)
                w = {k: project(v, M, mfl) for k, v in W[(t, s)].items()}; R = rel[t]; ids = top4[t]
                ens = np.mean([w[f"m{m}"] for m in HALF], 0)
                U = {m: float(util(w[f"m{m}"], R, gam, h)[0][0]) for m in HALF}
                u_ref = float(util(ew, R, gam, h)[0][0]); u_ens = float(util(ens, R, gam, h)[0][0])
                grid = util(GRID[:, None] * ew[None, :] + (1 - GRID)[:, None] * ens[None, :], R, gam, h)[0]
                q = ftrl_q(bank.sumG, bank.n, bank.sig, 0.9, 4.0)
                out = bank.decide(t, ids, U, u_ref, u_ens, grid)[name]; op = bool(bank.log[name][-1][1])
                if op:
                    sc = {m: np.mean(bank.G[m].scores) if len(bank.G[m].scores) >= MT.MIN_SCORES else -np.inf for m in ("identity", "content")}
                    lm = "content" if sc["content"] > sc["identity"] else "identity"
                    v = [bank.L[lm].predict(hh) for hh in ids]
                    best = max(HALF, key=lambda k: (sum(v[j] for j in range(len(ids)) if (k >> j) & 1), -bin(k).count("1")))
                    X = w[f"m{best}"]
                else:
                    X = q * ew + (1 - q) * ens
                chk = float(util(X, R, gam, h)[0][0])
                if abs(chk - out) > 1e-6: raise RuntimeError(f"deployed portfolio mismatch {name} {p} {s} {t}: {chk} vs {out}")
                fb.add(t, (t, ids, U, u_ref, u_ens, grid))
                res[di, pi, si] = out; cost[di, pi, si] = float(util(X, R, gam, h)[1][0]); opened[di, pi, si] = op; h = drift(X, nxt[t])
            fscore[f"{p}|{s}"] = {m: (float(np.mean(G.scores)) if G.scores else None, len(G.scores)) for m, G in bank.G.items()}
    return res, cost, opened, fscore


def paired_p(a, b):
    d = np.asarray(a) - np.asarray(b); n = len(d); sd = d.std(ddof=1)
    if sd == 0: return 0.0 if d.mean() > 0 else 1.0
    return MT.t_sf(d.mean() / (sd / math.sqrt(n)), n - 1)


def diag(C, R):
    W, seeds, dates, rel = C["W"], C["seeds"], C["dates"], C["rel"]; out = {}; nxt = to_next(dates)
    def spearman(a, b):
        ra = np.argsort(np.argsort(a)); rb = np.argsort(np.argsort(b)); return float(np.corrcoef(ra, rb)[0, 1])
    for pi, (p, (M, mfl, gam)) in enumerate(INV.items()):
        ew = project(np.full(K, 1.0 / K), M, mfl); h = ew.copy(); oc, cf, mk, det = [], [], [], 0
        for t in dates:
            U = np.array([[float(util(project(W[(t, s)][f"m{m}"], M, mfl), rel[t], gam, h)[0][0]) for m in FULL] for s in seeds])
            u1n = float(util(ew, rel[t], gam, h)[0][0]); Um = U.mean(0)
            for j in range(4):
                on = [m for m in FULL if (m >> j) & 1]; off = [m for m in FULL if not (m >> j) & 1]
                oc.append(Um[on].mean()); cf.append(Um[on].mean() - Um[off].mean()); mk.append(u1n)
            if len(seeds) > 1:
                grand = U.mean(); ssb = len(seeds) * ((Um - grand) ** 2).sum(); ssw = ((U - Um) ** 2).sum()
                dfw = 16 * (len(seeds) - 1); F = (ssb / 15) / (ssw / dfw) if ssw > 0 else np.inf; det += f_sf(F, 15, dfw) < 0.05
            h = drift(ew, nxt[t])
        ex = {k: np.nanmean(R[k][:, pi, :]) for k in EXPERIENCE}; best = max(ex, key=ex.get); n1 = np.nanmean(R["1/N"][:, pi, :])
        out[p] = dict(outcome_vs_1N=spearman(oc, mk), cf_vs_1N=spearman(cf, mk), detectable_months=int(det), months=len(dates),
                      best_agent=best, best_minus_1N_pp=100 * (ex[best] - n1),
                      cfsel_minus_1N_pp=100 * (np.nanmean(R["Counterfactual selection"][:, pi, :]) - n1),
                      method_minus_1N_pp=100 * (np.nanmean(R["MemTrial"][:, pi, :]) - n1))
    return out


def summarize(R, cost, dates, mask=None):
    sel = np.ones(len(dates), bool) if mask is None else mask; rows = {}
    for k, A in R.items():
        per_seed = 100 * np.nanmean(A[sel], (0, 1))
        rows[k] = dict(mean=float(per_seed.mean()), sd=float(per_seed.std(ddof=1)) if len(per_seed) > 1 else 0.0,
                       cost_pp=float(100 * np.nanmean(cost[k][sel])))
    by_date = {k: np.nanmean(A[sel], (1, 2)) for k, A in R.items()}
    best = max(EXPERIENCE, key=lambda k: rows[k]["mean"])
    return dict(rows=rows, best_experience_agent=best, improv_vs_best_pp=rows["MemTrial"]["mean"] - rows[best]["mean"],
                p_vs_best=paired_p(by_date["MemTrial"], by_date[best]),
                method_minus_1N_pp=rows["MemTrial"]["mean"] - rows["1/N"]["mean"], months=int(sel.sum()))


def main():
    mock = "--mock" in sys.argv
    name = sys.argv[sys.argv.index("--run") + 1] if "--run" in sys.argv else ("run_mock" if mock else "run")
    run = HERE / "classalloc" / name; ev = run / "eval"; ev.mkdir(exist_ok=True); t0 = time.time()
    stage = sys.argv[sys.argv.index("--stage") + 1] if "--stage" in sys.argv else "all"   # cache | base | mt0 | mt1 | mt2 | summary
    if stage == "cache" or (stage == "all" and (run / "executions").exists()): C = build_cache(run)   # otherwise eval/cache.pkl
    else: C = pickle.load(open(ev / "cache.pkl", "rb"))
    if stage in ("all", "base"):
        res, cost, held = evaluate_base(C); pickle.dump((res, cost, held), open(ev / "base.pkl", "wb"))
    for i, n in enumerate(MTV):
        if stage in ("all", f"mt{i}"): pickle.dump(evaluate_method(C, n), open(ev / f"mt{i}.pkl", "wb"))
    if stage not in ("all", "summary"): print(f"stage {stage} done ({time.time() - t0:.0f}s)"); return
    res, cost, held = pickle.load(open(ev / "base.pkl", "rb")); R = dict(res); Cst = dict(cost); trust = {}; fwd = {}
    for i, n in enumerate(MTV):
        r, c, op, fs = pickle.load(open(ev / f"mt{i}.pkl", "rb")); R[n] = r; Cst[n] = c; trust[n] = 100 * float(op.mean())
        sc = [v[m][0] for v in fs.values() for m in v if v[m][0] is not None]; fwd[n] = float(np.mean(sc)) if sc else None
    dates = C["dates"]; post = np.array([t >= "2024-07-01" for t in dates])
    out = {"run": name, "manifest": json.load(open(run / "MANIFEST.json")) if (run / "MANIFEST.json").exists() else None,
           "months": len(dates), "seeds": C["seeds"], "drafts_valid": C["valid"], "drafts_invalid": C["invalid"],
           "missing_baseline_draft_held": held, "all": summarize(R, Cst, dates),
           "before_cutoff": summarize(R, Cst, dates, ~post) if (~post).sum() > 2 else None,
           "after_cutoff": summarize(R, Cst, dates, post) if post.sum() > 2 else None,
           "trust_rate_pct": trust, "mean_forward_score": fwd, "table1": diag(C, R),
           "res_pp": {k: (100 * v).tolist() for k, v in R.items()}, "dates": dates}
    (run / "eval" / "RESULTS.json").write_text(json.dumps(out, indent=1))
    A = out["all"]
    print(f"\nClassAlloc ({name}): utility, pp per month (mean ± sd over seeds), {len(dates)} test months")
    for k in BASE + list(MTV): print(f"  {k:30s} {A['rows'][k]['mean']:7.3f} ± {A['rows'][k]['sd']:.3f}   cost {A['rows'][k]['cost_pp']:.3f}")
    print(f"  best experience-learning agent: {A['best_experience_agent']}; method - best = {A['improv_vs_best_pp']:+.3f} pp "
          f"(p = {A['p_vs_best']:.3f}); method - 1/N = {A['method_minus_1N_pp']:+.3f} pp; trust rate {trust['MemTrial']:.1f}%")
    for part in ("before_cutoff", "after_cutoff"):
        P = out[part]
        if P: print(f"  {part}: {P['months']} months; method - 1/N {P['method_minus_1N_pp']:+.3f} pp; best agent {P['best_experience_agent']} "
                    f"- 1/N {P['rows'][P['best_experience_agent']]['mean'] - P['rows']['1/N']['mean']:+.3f} pp; method - best {P['improv_vs_best_pp']:+.3f} (p = {P['p_vs_best']:.3f})")
    print(f"({time.time() - t0:.0f}s)")


if __name__ == "__main__":
    main()
