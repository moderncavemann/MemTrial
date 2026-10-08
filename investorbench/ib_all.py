"""All methods on the InvestorBench multi-asset extension, per seed (usage: python3 ib_all.py [run_dir]) -> ib_all.json
{"res": {method: [dates][investors][seeds]}, "dates": [...]}. Settlement: weights held from the close of day t to the close
of day t+1; mandate = risky cap M on the four stocks and cash floor m (scaling rule); fee 15 bps on the distance from the
equal-weight book; utility = r - (gamma/2) r^2 (r = daily net return), reported in percentage points per day.
MemGate = frozen version (../memtrial/MEMGATE_FROZEN.json): half-fraction designed subsets, content features = 8-dim PCA
of the experiences' text-embedding-3-small vectors."""
import sys, json, glob, math, random, collections
from pathlib import Path
import numpy as np
HERE = Path(__file__).resolve().parent; sys.path.insert(0, str(HERE)); sys.path.insert(0, str(HERE.parent / "memtrial"))
import run_ib as RB
from memgate import MemGateBank, banzhaf, GRID
HALF = [0, 3, 5, 6, 9, 10, 12, 15]
INV = {"conservative": (0.40, 0.40, 10.0), "balanced": (0.65, 0.20, 5.0), "aggressive": (0.90, 0.05, 2.0)}
FEE = 0.0015; KEYS = RB.KEYS
VARIANTS = {"MemGate": dict(learner="auto"), "MemGate | content learner only": dict(learner="content"),
            "MemGate | identity learner": dict(learner="identity"), "MemGate | F-test gate": dict(learner="auto", gate="ftest"),
            "MemGate | no gate": dict(learner="auto", gate="none"), "MemGate | closed->ensemble": dict(learner="auto", closed="ens"),
            "MemGate | closed->reference": dict(learner="auto", closed="ref"), "MemGate | uniform prior": dict(learner="auto", closed="uniform"),
            "MemGate | gate 0.2": dict(learner="auto", alpha=0.2), "MemGate | gate 0.01": dict(learner="auto", alpha=0.01)}


def vec(w): return np.array([w[k] for k in KEYS], float)


def project(x, M, m):
    x = np.maximum(np.asarray(x, float), 0); x = x / x.sum(); s = x[:4].sum(); cap = min(M, 1.0 - m)
    if s > cap: x[:4] *= cap / s; x[4] = 1.0 - x[:4].sum()
    return x


def utility(X, r, gamma):
    """X: (n, 5) projected weights; r: next-day returns of the 4 stocks (fraction)."""
    X = np.atleast_2d(X); prev = np.full(5, 0.2)
    net = X[:, :4] @ r - FEE * np.abs(X - prev).sum(1)
    return net - 0.5 * gamma * net ** 2


def main():
    out_dir = HERE / (sys.argv[1] if len(sys.argv) > 1 else "run")
    B = RB.Bench()
    recs = [json.load(open(f)) for f in glob.glob(str(out_dir / "executions" / "*.json"))]
    exps = json.load(open(out_dir / "experiences.json")); E = json.load(open(out_dir / "embeddings.json"))
    ids_all = [e["id"] for e in exps]; Xe = np.array([E["experiences"][h] for h in ids_all]); Xe = Xe - Xe.mean(0)
    _, _, Vt = np.linalg.svd(Xe, full_matrices=False); Zc = Xe @ Vt[:8].T; Zc = Zc / (Zc.std(0) + 1e-12); Z = {h: list(z) for h, z in zip(ids_all, Zc)}
    W = collections.defaultdict(dict); top4 = {}
    for r in recs:
        if r.get("phase") != "test" or r.get("status") != "VALID": continue
        arm = f"m{r['mask']}" if r["arm"] == "subset" else r["arm"]
        W[(r["date"], r["seed"])][arm] = vec(r["weights"])
        if r["arm"] == "subset": top4[r["date"]] = r["top4"]
    seeds = sorted({s for _, s in W}); dates = [t for t in B.test if all((t, s) in W and all(f"m{m}" in W[(t, s)] for m in range(16)) for s in seeds)]
    print(f"{len(dates)} complete test days, seeds {seeds}, failed executions {sum(1 for r in recs if r.get('status') != 'VALID')}")
    ret = {t: np.array([B.next_returns(t)[a] / 100.0 for a in RB.ASSETS]) for t in dates}
    past = {t: np.array([[B.price[a][B.days[i]] / B.price[a][B.days[i - 1]] - 1 for a in RB.ASSETS] for i in range(B.idx[t] - 59, B.idx[t] + 1)]) for t in dates}
    res = collections.defaultdict(lambda: np.full((len(dates), 3, len(seeds)), np.nan))
    for pi, (p, (M, mfl, gam)) in enumerate(INV.items()):
        for si, s in enumerate(seeds):
            bank = MemGateBank(Z, VARIANTS, masks=HALF); hist = []; up_obs = []; hed = []; rnd = random.Random(f"{p}|{s}")
            for di, t in enumerate(dates):
                w = {k: project(v, M, mfl) for k, v in W[(t, s)].items()}; r = ret[t]
                ew = project(np.full(5, 0.2), M, mfl)
                C = np.cov(past[t].T); x = np.full(4, 0.25); Lc = 2 * np.linalg.eigvalsh(C).max()
                for _ in range(300): x = x - (2 * C @ x) / Lc; x = np.maximum(x, 0); x = x / x.sum()
                mv = project(np.append(x * min(M, 1 - mfl), 1 - min(M, 1 - mfl)), M, mfl)
                sc_vec = project(np.mean([W[(t, ss)]["m0"] for ss in seeds], 0), M, mfl)
                ens8 = np.mean([w[f"m{m}"] for m in HALF], 0)
                arms = {**{k: w[k] for k in w}, "1/N": ew, "MinVar": mv, "SC": sc_vec, "ENS8": ens8}
                names = list(arms); u = dict(zip(names, utility(np.array([arms[k] for k in names]), r, gam)))
                ids = top4[t]
                o = {"1/N": u["1/N"], "Minimum variance": u["MinVar"], "Zero-shot (no memory)": u["m0"], "Self-consistency": u["SC"],
                     "Similarity retrieval (top-2)": u["m3"], "Similarity retrieval (top-4)": u["m15"], "FinMem": u.get("finmem", np.nan),
                     "MemRL": u.get("memrl", np.nan), "Reflexion": u.get("reflexion", np.nan), "ExpeL": u.get("expel", np.nan),
                     "Draft averaging": u["ENS8"], "Oracle": max(u[k] for k in w)}
                if hist:
                    cf = collections.defaultdict(list)
                    for h in hist:
                        for j, hid in enumerate(h["ids"]): cf[hid].append(h["cf"][j])
                    scs = [np.mean(cf[h]) if h in cf else 0.0 for h in ids]; od = sorted(range(4), key=lambda j: (-scs[j], j))
                    o["Counterfactual selection"] = u[f"m{(1 << od[0]) | (1 << od[1])}"]
                else:
                    o["Counterfactual selection"] = u["m3"]
                if len(up_obs) >= 3:
                    allid = sorted({h for q in up_obs for h in q[0]} | set(ids)); ix = {h: i for i, h in enumerate(allid)}
                    Xr = np.array([[1.0, q[2]] + [1.0 if h in q[0] else 0.0 for h in allid] for q in up_obs]); yr = np.array([q[1] for q in up_obs])
                    pen = np.eye(Xr.shape[1]); pen[0, 0] = pen[1, 1] = 0.0
                    coef = np.linalg.solve(Xr.T @ Xr + 1e-4 * pen * len(yr), Xr.T @ yr); upl = [coef[2 + ix[h]] for h in ids]
                else:
                    upl = [0.0] * 4
                od = rnd.sample(range(4), 2) if rnd.random() < 0.1 else sorted(range(4), key=lambda j: (-upl[j], j))[:2]
                mk = f"m{(1 << od[0]) | (1 << od[1])}"; o["Uplift credit"] = u[mk]; up_obs.append(({ids[od[0]], ids[od[1]]}, u[mk], u["1/N"]))
                Wf = np.array([ew, w["m0"], w["m15"], w.get("finmem", w["m0"]), w.get("memrl", w["m0"])]); uf = utility(Wf, r, gam)
                if hed:
                    Uh = np.array(hed); sg = float(np.mean(Uh.std(1))) + 1e-12; lam = 4.0 * sg / math.sqrt(len(Uh)); zz = Uh.mean(0) / lam
                    qh = np.exp(zz - zz.max()); qh /= qh.sum()
                else:
                    qh = np.full(5, 0.2)
                o["Hedge"] = float(utility((qh @ Wf)[None, :], r, gam)[0]); hed.append(list(uf))
                Uh = {m: u[f"m{m}"] for m in HALF}; grid = utility(GRID[:, None] * ew[None, :] + (1 - GRID)[:, None] * ens8[None, :], r, gam)
                o.update(bank.decide(t, ids, Uh, u["1/N"], u["ENS8"], grid))
                for k, v in o.items(): res[k][di, pi, si] = v
                bank.matured(t, ids, Uh, u["1/N"], u["ENS8"], grid)
                hist.append({"ids": ids, "cf": [banzhaf({m: u[f"m{m}"] for m in range(16)}, j, list(range(16))) for j in range(4)]})
    J = {"res": {k: np.where(np.isnan(v), None, v).tolist() for k, v in res.items()}, "dates": dates, "seeds": seeds}
    (HERE / ("ib_all.json" if out_dir.name == "run" else f"ib_all_{out_dir.name}.json")).write_text(json.dumps(J))
    for k in ["1/N", "Zero-shot (no memory)", "FinMem", "MemRL", "Reflexion", "ExpeL", "Hedge", "MemGate"]:
        x = np.array(res[k], float).mean((0, 1)); print(f"{k:28s} {100*np.nanmean(x):.4f} +- {100*np.nanstd(x, ddof=1):.4f} (pp/day)")


if __name__ == "__main__":
    main()
