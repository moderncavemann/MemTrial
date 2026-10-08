"""InvestorBench with the fee charged on actual trades (usage: python3 ib_turnover.py STAGE [NAMES]).

ib_all.py charges the 15 bp fee every day on the L1 distance between the day's portfolio and the equal-weight book
(0.2 in each stock and in cash), whether or not anything is traded: a conservative investor who simply holds the
mandate-projected 1/N portfolio (0.1 per stock, 0.6 cash) pays 12 bp per day. Here the fee is charged on what is traded:
the L1 change between the holdings a method carries into the day (its previous portfolio after one day of price drift)
and its new target. Everything else is as in ib_all.py: same logged drafts, mandates, utility, methods and MemGate code
(memgate.py frozen, unmodified). Every method starts the test period holding the mandate-projected 1/N portfolio and carries
its own holdings forward, so the fee of each candidate it considers depends on what it holds; its learning signals
(counterfactual contributions, uplift regression, Hedge feedback, the FTRL statistics of the anchored allocation) use the
same fee as its evaluation. Published files (ib_all.json, IB_DIAG.json) are not touched.

Stages:
  cache               -> turnover/cache.pkl (drafts, returns, retrieval, content features)
  base                -> turnover/base.json (all methods except MemGate)
  mg NAME[;NAME...]   -> turnover/mg_<k>.json, one MemGate variant per name (keys of VARIANTS)
  merge               -> ib_all_turnover.json (layout of ib_all.json), turnover/SUMMARY.json
  diag                -> IB_DIAG_turnover.json (Table 1 rows)"""
import sys, json, glob, math, random, collections, pickle, time, re
from pathlib import Path
import numpy as np
HERE = Path(__file__).resolve().parent; sys.path.insert(0, str(HERE)); sys.path.insert(0, str(HERE.parent / "memtrial"))
import memgate as MG
from memgate import MemGateBank, banzhaf, GRID, ftrl_q, f_sf
OUT = HERE / "turnover"; OUT.mkdir(exist_ok=True)
HALF = [0, 3, 5, 6, 9, 10, 12, 15]; FULL = list(range(16))
INV = {"conservative": (0.40, 0.40, 10.0), "balanced": (0.65, 0.20, 5.0), "aggressive": (0.90, 0.05, 2.0)}
FEE = 0.0015
EXPERIENCE = ("FinMem", "MemRL", "Reflexion", "ExpeL")

VARIANTS = {"MemGate": dict(learner="auto"), "MemGate | content learner only": dict(learner="content"),
            "MemGate | identity learner": dict(learner="identity"), "MemGate | F-test gate": dict(learner="auto", gate="ftest"),
            "MemGate | no gate": dict(learner="auto", gate="none"), "MemGate | closed->ensemble": dict(learner="auto", closed="ens"),
            "MemGate | closed->reference": dict(learner="auto", closed="ref"), "MemGate | uniform prior": dict(learner="auto", closed="uniform"),
            "MemGate | gate 0.2": dict(learner="auto", alpha=0.2), "MemGate | gate 0.01": dict(learner="auto", alpha=0.01),
            "S|alpha=0.02": dict(learner="auto", alpha=0.02), "S|alpha=0.1": dict(learner="auto", alpha=0.1),
            "S|minscores=5": dict(learner="auto", minsc=5), "S|minscores=20": dict(learner="auto", minsc=20),
            "S16|MemGate": dict(learner="auto", full=True), "S16|no gate": dict(learner="auto", gate="none", full=True)}
for _a0 in (0.5, 0.75, 0.9, 0.95):
    for _l0 in (1.0, 4.0, 16.0):
        if (_a0, _l0) != (0.9, 4.0): VARIANTS[f"S|a0={_a0},lam0={_l0}"] = dict(learner="auto", a0=_a0, lam0=_l0)


def slug(name): return re.sub(r"[^A-Za-z0-9]+", "_", name).strip("_")


def project(x, M, m):                       # identical to ib_all.project
    x = np.maximum(np.asarray(x, float), 0); x = x / x.sum(); s = x[:4].sum(); cap = min(M, 1.0 - m)
    if s > cap: x[:4] *= cap / s; x[4] = 1.0 - x[:4].sum()
    return x


def drift(x, r):
    """weights after one day of price moves (cash earns nothing, as in ib_all)."""
    g = np.asarray(x, float) * np.append(1.0 + r, 1.0); return g / g.sum()


def parts(X, r, h):
    X = np.atleast_2d(X); gross = X[:, :4] @ r; cost = FEE * np.abs(X - h).sum(1); return gross, cost


def util(X, r, gam, h):
    """utility of target(s) X for a method holding h at the start of the day; fee on the traded amount |X - h|."""
    gross, cost = parts(X, r, h); net = gross - cost
    return net - 0.5 * gam * net ** 2


def minvar(past_t, M, mfl):                 # identical to ib_all
    C = np.cov(past_t.T); x = np.full(4, 0.25); Lc = 2 * np.linalg.eigvalsh(C).max()
    for _ in range(300): x = x - (2 * C @ x) / Lc; x = np.maximum(x, 0); x = x / x.sum()
    return project(np.append(x * min(M, 1 - mfl), 1 - min(M, 1 - mfl)), M, mfl)


def stage_cache():
    import run_ib as RB, ib_all as IA
    B = RB.Bench(); run = HERE / "run"
    if not (run / "executions").exists(): sys.exit("turnover/cache.pkl is built from the logs of run_ib.py (run/executions), which are not included")
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
    pickle.dump(dict(W=dict(W), top4=top4, seeds=seeds, dates=dates, ret=ret, past=past, Z=Z), open(OUT / "cache.pkl", "wb"))
    print(len(dates), "test days; seeds", seeds, "; missing baseline arms:", dict(miss))


def load(): return pickle.load(open(OUT / "cache.pkl", "rb"))


def stage_base():
    C = load(); W, top4, seeds, dates, ret, past = C["W"], C["top4"], C["seeds"], C["dates"], C["ret"], C["past"]
    nD, nS = len(dates), len(seeds)
    NAMES = ["1/N", "Minimum variance", "Zero-shot (no memory)", "Self-consistency", "Similarity retrieval (top-2)",
             "Similarity retrieval (top-4)", "FinMem", "MemRL", "Reflexion", "ExpeL", "Draft averaging",
             "Counterfactual selection", "Uplift credit", "Hedge"]
    res = {k: np.full((nD, 3, nS), np.nan) for k in NAMES}; cost = {k: np.full((nD, 3, nS), np.nan) for k in NAMES}
    gross = {k: np.full((nD, 3, nS), np.nan) for k in NAMES}; held = collections.Counter()
    ARM = {"FinMem": "finmem", "MemRL": "memrl", "Reflexion": "reflexion", "ExpeL": "expel"}
    for pi, (p, (M, mfl, gam)) in enumerate(INV.items()):
        ew = project(np.full(5, 0.2), M, mfl)
        mv = {t: minvar(past[t], M, mfl) for t in dates}
        sc = {t: project(np.mean([W[(t, ss)]["m0"] for ss in seeds], 0), M, mfl) for t in dates}
        for si, s in enumerate(seeds):
            H = {k: ew.copy() for k in NAMES}; hist = []; up_obs = []; hed = []; rnd = random.Random(f"{p}|{s}")
            for di, t in enumerate(dates):
                w = {k: project(v, M, mfl) for k, v in W[(t, s)].items()}; r = ret[t]; ids = top4[t]
                ens8 = np.mean([w[f"m{m}"] for m in HALF], 0)
                target = {"1/N": ew, "Minimum variance": mv[t], "Zero-shot (no memory)": w["m0"], "Self-consistency": sc[t],
                          "Similarity retrieval (top-2)": w["m3"], "Similarity retrieval (top-4)": w["m15"], "Draft averaging": ens8}
                for k, a in ARM.items():
                    if a in w: target[k] = w[a]
                    else: target[k] = H[k]; held[k] += 1          # missing draft: keep yesterday's holdings (no trade)
                # counterfactual selection: top-2 by mean past Banzhaf contribution (contributions under its own holdings)
                hC = H["Counterfactual selection"]; u16 = {m: float(util(w[f"m{m}"], r, gam, hC)[0]) for m in FULL}
                if hist:
                    cf = collections.defaultdict(list)
                    for hh in hist:
                        for j, hid in enumerate(hh["ids"]): cf[hid].append(hh["cf"][j])
                    scs = [np.mean(cf[h]) if h in cf else 0.0 for h in ids]; od = sorted(range(4), key=lambda j: (-scs[j], j))
                    target["Counterfactual selection"] = w[f"m{(1 << od[0]) | (1 << od[1])}"]
                else:
                    target["Counterfactual selection"] = w["m3"]
                hist.append({"ids": ids, "cf": [banzhaf(u16, j, FULL) for j in range(4)]})
                # uplift credit: regression of realised utility on chosen-experience indicators, adjusted for the 1/N utility
                if len(up_obs) >= 3:
                    allid = sorted({h for q in up_obs for h in q[0]} | set(ids)); ix = {h: i for i, h in enumerate(allid)}
                    Xr = np.array([[1.0, q[2]] + [1.0 if h in q[0] else 0.0 for h in allid] for q in up_obs]); yr = np.array([q[1] for q in up_obs])
                    pen = np.eye(Xr.shape[1]); pen[0, 0] = pen[1, 1] = 0.0
                    coef = np.linalg.solve(Xr.T @ Xr + 1e-4 * pen * len(yr), Xr.T @ yr); upl = [coef[2 + ix[h]] for h in ids]
                else:
                    upl = [0.0] * 4
                od = rnd.sample(range(4), 2) if rnd.random() < 0.1 else sorted(range(4), key=lambda j: (-upl[j], j))[:2]
                mk = f"m{(1 << od[0]) | (1 << od[1])}"; target["Uplift credit"] = w[mk]
                # Hedge over five memory families; feedback = utility of each family under Hedge's own holdings
                Wf = np.array([ew, w["m0"], w["m15"], w.get("finmem", w["m0"]), w.get("memrl", w["m0"])])
                uf = util(Wf, r, gam, H["Hedge"])
                if hed:
                    Uh = np.array(hed); sg = float(np.mean(Uh.std(1))) + 1e-12; lam = 4.0 * sg / math.sqrt(len(Uh)); zz = Uh.mean(0) / lam
                    qh = np.exp(zz - zz.max()); qh /= qh.sum()
                else:
                    qh = np.full(5, 0.2)
                target["Hedge"] = qh @ Wf; hed.append(list(uf))
                for k in NAMES:
                    X = target[k]; g, c = parts(X, r, H[k]); u = float(util(X, r, gam, H[k])[0])
                    res[k][di, pi, si] = u; cost[k][di, pi, si] = float(c[0]); gross[k][di, pi, si] = float(g[0])
                    H[k] = drift(X, r)
                up_obs.append(({ids[od[0]], ids[od[1]]}, res["Uplift credit"][di, pi, si], res["1/N"][di, pi, si]))
    tl = lambda a: np.where(np.isnan(a), None, a).tolist()
    json.dump({"res": {k: tl(v) for k, v in res.items()}, "cost": {k: tl(v) for k, v in cost.items()},
               "gross": {k: tl(v) for k, v in gross.items()}, "held_missing": dict(held)}, open(OUT / "base.json", "w"))
    for k in NAMES:
        print(f"{k:30s} utility {1e4*np.nanmean(res[k]):7.2f}  cost {1e4*np.nanmean(cost[k]):5.2f}  gross {1e4*np.nanmean(gross[k]):6.2f} bp/day")
    print("days a missing draft was replaced by holding:", dict(held))


def run_variant(name, C):
    cfg = dict(VARIANTS[name]); full = cfg.pop("full", False); a0 = cfg.pop("a0", 0.9); lam0 = cfg.pop("lam0", 4.0); ms = cfg.pop("minsc", None)
    masks = FULL if full else HALF; cl = cfg.get("closed", "ftrl")
    W, top4, seeds, dates, ret, Z = C["W"], C["top4"], C["seeds"], C["dates"], C["ret"], C["Z"]
    nD, nS = len(dates), len(seeds)
    res = np.full((nD, 3, nS), np.nan); cost = np.full((nD, 3, nS), np.nan); opened = np.zeros((nD, 3, nS)); qs = np.full((nD, 3, nS), np.nan)
    fscore = {}
    keep = MG.MIN_SCORES
    if ms is not None: MG.MIN_SCORES = ms
    try:
        for pi, (p, (M, mfl, gam)) in enumerate(INV.items()):
            ew = project(np.full(5, 0.2), M, mfl)
            for si, s in enumerate(seeds):
                bank = MemGateBank(Z, {name: cfg}, masks=masks, a0=a0, lam0=lam0); h = ew.copy()
                for di, t in enumerate(dates):
                    w = {k: project(v, M, mfl) for k, v in W[(t, s)].items()}; r = ret[t]; ids = top4[t]
                    ens = np.mean([w[f"m{m}"] for m in masks], 0)
                    U = {m: float(util(w[f"m{m}"], r, gam, h)[0]) for m in masks}
                    u_ref = float(util(ew, r, gam, h)[0]); u_ens = float(util(ens, r, gam, h)[0])
                    grid = util(GRID[:, None] * ew[None, :] + (1 - GRID)[:, None] * ens[None, :], r, gam, h)
                    q = ftrl_q(bank.sumG, bank.n, bank.sig, 0.5 if cl == "uniform" else a0, lam0)
                    out = bank.decide(t, ids, U, u_ref, u_ens, grid)[name]
                    op = bool(bank.log[name][-1][1])
                    if op:                                  # the designed draft the bank deployed
                        lm = cfg.get("learner", "content")
                        if lm == "auto":
                            sc = {m: np.mean(bank.G[m].scores) if len(bank.G[m].scores) >= MG.MIN_SCORES else -np.inf for m in ("identity", "content")}
                            lm = "content" if sc["content"] > sc["identity"] else "identity"
                        v = [bank.L[lm].predict(hh) for hh in ids]
                        best = max(masks, key=lambda k: (sum(v[j] for j in range(len(ids)) if (k >> j) & 1), -bin(k).count("1")))
                        X = w[f"m{best}"]
                    elif cl == "ens": X = ens
                    elif cl == "ref": X = ew
                    else: X = q * ew + (1 - q) * ens; qs[di, pi, si] = q
                    chk = float(util(X, r, gam, h)[0])
                    if abs(chk - out) > 1e-10: raise RuntimeError(f"deployed portfolio mismatch {name} {p} {s} {t}: {chk} vs {out}")
                    bank.matured(t, ids, U, u_ref, u_ens, grid)
                    res[di, pi, si] = out; cost[di, pi, si] = float(parts(X, r, h)[1][0]); opened[di, pi, si] = op; h = drift(X, r)
                fscore[f"{p}|{s}"] = {m: (float(np.mean(G.scores)) if G.scores else None, len(G.scores)) for m, G in bank.G.items()}
    finally:
        MG.MIN_SCORES = keep
    json.dump({"name": name, "res": res.tolist(), "cost": cost.tolist(), "open": opened.tolist(),
               "q": np.where(np.isnan(qs), None, qs).tolist(), "forward_score": fscore}, open(OUT / f"mg_{slug(name)}.json", "w"))
    return res, opened


def stage_mg(names):
    C = load()
    for n in names:
        t0 = time.time(); res, op = run_variant(n, C)
        print(f"{n:32s} {1e4*np.mean(res):6.2f} bp/day  open {100*op.mean():5.2f}%  ({time.time()-t0:.0f}s)", flush=True)


def stage_merge():
    C = load(); B = json.load(open(OUT / "base.json")); res = dict(B["res"]); sens = {}; opn = {}; fsc = {}; q = {}
    missing = [n for n in VARIANTS if not (OUT / f"mg_{slug(n)}.json").exists()]
    if missing: raise SystemExit(f"missing variants: {missing}")
    for n in VARIANTS:
        J = json.load(open(OUT / f"mg_{slug(n)}.json")); a = np.array(J["res"], float)
        opn[n] = float(np.mean(J["open"])); fsc[n] = J["forward_score"]
        qq = np.array(J["q"], float); q[n] = float(np.nanmean(qq)) if np.isfinite(qq).any() else None
        if n.startswith("MemGate"): res[n] = J["res"]
        sens[n] = (1e4 * a.mean((0, 1))).tolist()                    # per-seed mean, bp/day
    out = {"res": res, "dates": C["dates"], "seeds": C["seeds"],
           "fee_rule": "15 bps on the L1 change between the drifted holdings carried into the day and the new target"}
    (HERE / "ib_all_turnover.json").write_text(json.dumps(out))
    cost = {k: 1e4 * float(np.nanmean(np.array(v, float))) for k, v in B["cost"].items()}
    cost["MemGate"] = 1e4 * float(np.nanmean(np.array(json.load(open(OUT / f"mg_{slug('MemGate')}.json"))["cost"], float)))
    summ = {"per_seed_bp": sens, "open_share": opn, "mean_q_closed": q, "forward_score": fsc, "mean_cost_bp": cost,
            "mean_gross_bp": {k: 1e4 * float(np.nanmean(np.array(v, float))) for k, v in B["gross"].items()}}
    (OUT / "SUMMARY.json").write_text(json.dumps(summ, indent=1))
    A = {k: np.array(v, float) for k, v in res.items()}
    for k in ["1/N", "Zero-shot (no memory)", "FinMem", "MemRL", "Reflexion", "ExpeL", "Counterfactual selection", "Hedge", "Draft averaging", "MemGate", "MemGate | no gate"]:
        x = np.nanmean(A[k], (0, 1)); print(f"{k:28s} {1e4*x.mean():6.2f} ± {1e4*x.std(ddof=1):.2f} bp/day   cost {cost.get(k, float('nan')):.2f}")
    print("open share MemGate:", round(100 * opn["MemGate"], 2), "%")


def stage_diag():
    C = load(); W, seeds, dates, ret = C["W"], C["seeds"], C["dates"], C["ret"]
    J = json.load(open(HERE / "ib_all_turnover.json")); R = {k: np.array(v, float) for k, v in J["res"].items()}
    def spearman(a, b):
        ra = np.argsort(np.argsort(a)); rb = np.argsort(np.argsort(b)); return float(np.corrcoef(ra, rb)[0, 1])
    out = {}
    for pi, (p, (M, mfl, gam)) in enumerate(INV.items()):
        ew = project(np.full(5, 0.2), M, mfl); h = ew.copy()           # common book for the day: the 1/N holdings
        oc, cf, mk, det = [], [], [], 0
        for t in dates:
            r = ret[t]
            U = np.array([[float(util(project(W[(t, s)][f"m{m}"], M, mfl), r, gam, h)[0]) for m in FULL] for s in seeds])
            u1n = float(util(ew, r, gam, h)[0]); Um = U.mean(0)
            for j in range(4):
                on = [m for m in FULL if (m >> j) & 1]; off = [m for m in FULL if not (m >> j) & 1]
                oc.append(Um[on].mean()); cf.append(Um[on].mean() - Um[off].mean()); mk.append(u1n)
            grand = U.mean(); ssb = len(seeds) * ((Um - grand) ** 2).sum(); ssw = ((U - Um) ** 2).sum()
            dfw = 16 * (len(seeds) - 1); F = (ssb / 15) / (ssw / dfw) if ssw > 0 else np.inf; det += f_sf(F, 15, dfw) < 0.05
            h = drift(ew, r)
        ex = {k: np.nanmean(R[k][:, pi, :]) for k in EXPERIENCE}; best = max(ex, key=ex.get); n1 = np.nanmean(R["1/N"][:, pi, :])
        out[p] = dict(outcome_vs_1N=spearman(oc, mk), cf_vs_1N=spearman(cf, mk), detectable_days=int(det), days=len(dates),
                      chance=0.05 * len(dates), best_agent=best, best_minus_1N_bp=1e4 * (ex[best] - n1),
                      cfsel_minus_1N_bp=1e4 * (np.nanmean(R["Counterfactual selection"][:, pi, :]) - n1),
                      memgate_minus_1N_bp=1e4 * (np.nanmean(R["MemGate"][:, pi, :]) - n1),
                      one_over_n_bp=1e4 * n1)
    (HERE / "IB_DIAG_turnover.json").write_text(json.dumps(out, indent=1)); print(json.dumps(out, indent=1))


if __name__ == "__main__":
    st = sys.argv[1]
    if st == "cache": stage_cache()
    elif st == "base": stage_base()
    elif st == "mg": stage_mg(sys.argv[2].split(";"))
    elif st == "merge": stage_merge()
    elif st == "diag": stage_diag()
    else: raise SystemExit(__doc__)
