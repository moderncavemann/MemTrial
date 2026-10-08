"""PlantedMem: semi-synthetic benchmark with planted experiences of known quality (offline; no LLM calls).

World (world.py, env.py): real PortBench prices aggregated into four asset-class indices (equities, bonds, cash,
commodities), one decision per month held for 20 sessions, and memory-free drafts resampled from 2,828 logged
gpt-4.1-mini PortBench allocations. Experiences: M = 12 (or 48) planted experiences, a quarter useful, half neutral and a
quarter misleading. With probability p_inf a useful (misleading) experience predicts the class with the highest (lowest)
next-month return, otherwise a random class; a draft that uses a set of experiences is tilted by beta towards their
predictions. Retrieval ranks experiences by a score that depends on the market regime, which reproduces the market
confound of outcome credit. Regimes (REGIMES): no influence, noise only, beta 0.1 / 0.25 / 0.5, and 48 experiences whose
content is uninformative or informative about their quality. Investors (GAMMA): risk aversion 2, 5 and 10.
Every method sees the same retrieved experiences; MemTrial is MemTrialBank of ../memtrial/memtrial.py (frozen) and runs
together with its ablation variants (MT_VARIANTS). The first `warm` dates are not scored.

Usage: python3 suite.py run|prun <investor> <s0> <s1>  -> results/suite5_<investor>_<s0>_<s1>.json
       investor in {aggressive, balanced, conservative}; dev seeds 0-14, test seeds 15-114 (the paper's tables);
       prun spreads the episodes over 4 processes and gives the same results as run."""
import sys, math, random, collections, json, time
from pathlib import Path
import numpy as np
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import world as DR
sys.path.insert(0, str(HERE.parent / "memtrial"))
from memtrial import MemTrialBank
MT_VARIANTS = {"MemTrial": dict(learner="auto"), "MemTrial | content learner only": dict(learner="content"),
               "MemTrial | identity learner": dict(learner="identity"), "MemTrial | F-test gate": dict(learner="auto", gate="ftest"),
               "MemTrial | no gate": dict(learner="auto", gate="none"), "MemTrial | closed->ensemble": dict(learner="auto", closed="ens"),
               "MemTrial | closed->reference": dict(learner="auto", closed="ref"), "MemTrial | uniform prior": dict(learner="auto", closed="uniform"),
               "MemTrial | gate 0.2": dict(learner="auto", alpha=0.2), "MemTrial | gate 0.01": dict(learner="auto", alpha=0.01)}
D, BASE, REL, RET, REG, HALF = DR.D, DR.BASE, DR.REL, DR.RET, DR.REG, DR.HALF
OUT = HERE / "results"; OUT.mkdir(exist_ok=True)
GRID = np.linspace(0.0, 1.0, 201)
GAMMA = {"aggressive": 2.0, "balanced": 5.0, "conservative": 10.0}
REGIMES = {"no influence": dict(beta=0.0), "noise only": dict(beta=0.5, noise_only=True), "beta 0.1": dict(beta=0.1),
           "beta 0.25": dict(beta=0.25), "beta 0.5": dict(beta=0.5),
           "many experiences": dict(beta=0.25, M=48), "many experiences, informative content": dict(beta=0.25, M=48, kappa=1.5)}


def U_batch(Wm, rel, gamma, fee=0.0015):
    """vectorised env.utility (same formula) for rows of Wm."""
    Wm = Wm / Wm.sum(1, keepdims=True); nav = 1 - fee * np.abs(Wm - 0.25).sum(1)
    path = np.concatenate([np.ones((len(Wm), 1)), nav[:, None] * (Wm @ rel)], 1); r = path[:, 1:] / path[:, :-1] - 1
    return path[:, -1] - 1 - 0.5 * gamma * r.var(1) * 20


def episode(beta, seed, M=12, warm=12, p_inf=0.7, noise_only=False, gamma=5.0, kappa=0.0):
    """One world (seed) of one regime for one investor -> ({method: mean utility over the scored dates}, MemTrial trust rate).
    Keys: the baselines of Table 3 (see ../tables/sd_pm.py and ../memtrial/paper_tables.py for the paper's names), MemTrial
    ('MemTrial'), its variants, and '_open|<variant>' (share of dates on which the variant acted on its learned values)."""
    U = lambda w, t: DR.env.utility(w, REL[t], gamma=gamma)
    rng = np.random.default_rng(seed); prng = random.Random(seed); rng2 = np.random.default_rng(seed + 10 ** 6)
    types = np.array([0] * M) if noise_only else np.array([1] * (M // 4) + [0] * (M - 2 * (M // 4)) + [-1] * (M // 4)); rng.shuffle(types)
    home = rng.integers(0, 2, M)                                 # retrieval home regime of each experience, independent of its type
    def predicted_class(m, t):
        if types[m] == 1 and rng.random() < p_inf: return int(np.argmax(RET[t]))
        if types[m] == -1 and rng.random() < p_inf: return int(np.argmin(RET[t]))
        return int(rng.integers(0, 4))
    def act(S, preds, base):
        if not S: return base
        tilt = np.mean([np.eye(4)[preds[m]] for m in S], 0); return (1 - beta) * base + beta * tilt
    Q = np.zeros(M); obs = collections.defaultdict(list); res = collections.defaultdict(list)
    up_obs = []; uprng = random.Random(seed + 7)
    rng5 = np.random.default_rng(seed + 4 * 10 ** 6)
    Zc = {int(m): list(kappa * types[m] * np.eye(8)[0] + rng5.normal(0, 1, 8)) for m in range(M)}     # content features
    MT = MemTrialBank(Zc, MT_VARIANTS, masks=HALF)
    rng6 = np.random.default_rng(seed + 5 * 10 ** 6)
    imp = np.full(M, 0.5); votes = np.zeros(M); refl = []; hed = []; pastret = []
    for t in range(len(D)):
        score = (home == REG[t]).astype(float) + rng.normal(0, 0.5, M); ret4 = list(np.argsort(-score)[:4])
        preds = {m: predicted_class(m, t) for m in ret4}
        b = lambda: BASE[rng.integers(len(BASE))]
        sub = lambda mask: [ret4[j] for j in range(4) if (mask >> j) & 1]
        out = {}
        out["no memory"] = U(b(), t)
        out["similarity top-2"] = U(act(ret4[:2], preds, b()), t)
        out["all 4"] = U(act(ret4, preds, b()), t)
        best_true = sorted(ret4, key=lambda m: (-types[m], ret4.index(m)))[:2]
        out["oracle (knows quality)"] = U(act([m for m in best_true if types[m] == 1] or [], preds, b()), t)   # not reported
        if prng.random() < 0.1: used = prng.sample(ret4, 2)
        else: used = sorted(ret4, key=lambda m: (-Q[m], ret4.index(m)))[:2]
        u = U(act(used, preds, b()), t); out["outcome credit top-2 (MemRL/FinMem-style)"] = u
        for m in used: Q[m] += 0.2 * (u - Q[m])
        mem = {mask: act(sub(mask), preds, b()) for mask in HALF}; umem = {mask: U(w, t) for mask, w in mem.items()}
        mu = np.zeros(4)
        for j, m in enumerate(ret4):
            v = obs[m]
            if len(v) >= 2: mu[j] = np.mean(v)
        top = sorted(range(4), key=lambda j: (-mu[j], j))[:2]
        out["counterfactual credit argmax top-2"] = U(act([ret4[j] for j in top], preds, b()), t)
        ens_vec = np.mean([mem[mask] for mask in HALF], 0)
        out["uniform aggregation of the 8 members"] = U(ens_vec, t)
        out["similarity top-2, 8-run self-consistency"] = U(np.mean([act(ret4[:2], preds, b()) for _ in range(8)], 0), t)   # not reported
        ref_vec = np.mean([BASE[rng2.integers(len(BASE))] for _ in range(8)], 0)
        refavg = U(ref_vec, t)
        out["no-memory 8-draw ensemble (reference)"] = refavg
        ens = out["uniform aggregation of the 8 members"]
        if len(up_obs) >= 3:
            Xr = np.array([[1.0, o[2]] + [1.0 if m in o[0] else 0.0 for m in range(M)] for o in up_obs]); yr = np.array([o[1] for o in up_obs])
            pen = np.eye(Xr.shape[1]); pen[0, 0] = pen[1, 1] = 0.0
            coef = np.linalg.solve(Xr.T @ Xr + 1e-4 * pen * len(yr), Xr.T @ yr)
            upl = {m: coef[2 + m] for m in range(M)}
        else:
            upl = {m: 0.0 for m in range(M)}
        if uprng.random() < 0.1: up_used = uprng.sample(list(ret4), 2)
        else: up_used = sorted(ret4, key=lambda m: (-upl[m], ret4.index(m)))[:2]
        u_up = U(act(up_used, preds, BASE[rng2.integers(len(BASE))]), t)
        out["uplift credit top-2 (regression-adjusted, UpliftMem/UCOB-style)"] = u_up
        up_obs.append((set(int(m) for m in up_used), u_up, U(np.ones(4) / 4, t)))
        rel = REL[t]
        gridE = U_batch(GRID[:, None] * ref_vec[None, :] + (1 - GRID)[:, None] * ens_vec[None, :], rel, gamma)
        # ---- adapted baselines (selection among the 4 retrieved experiences uses the executed pair members) ----
        ew = np.ones(4) / 4; u_ew = float(U_batch(ew[None, :], rel, gamma)[0]); out["1/N"] = u_ew
        if len(pastret) >= 20:
            Rp = np.array(pastret[-60:]); C = np.cov(Rp.T) * 20; x = ew.copy(); Lc = 2 * np.linalg.eigvalsh(C).max()
            for _ in range(200):
                x = x - (2 * C @ x) / Lc; x = np.maximum(x, 0); x = x / x.sum() if x.sum() > 0 else ew.copy()
            out["Minimum variance"] = float(U_batch(x[None, :], rel, gamma)[0])
        else:
            out["Minimum variance"] = u_ew
        pos = {m: j for j, m in enumerate(ret4)}
        pair = lambda a_, b_: umem[(1 << pos[a_]) | (1 << pos[b_])]
        fm = sorted(ret4, key=lambda m: (-(score[m] + imp[m]), pos[m]))[:2]; out["FinMem (adapted)"] = u_fm = pair(*fm)
        xp = sorted(ret4, key=lambda m: (-votes[m], pos[m]))[:2]; out["ExpeL (adapted)"] = u_xp = pair(*xp)
        base6 = BASE[rng6.integers(len(BASE))]
        tilt = np.mean([np.eye(4)[k] for k in refl[-3:]], 0) if refl else None
        out["Reflexion (adapted)"] = float(U_batch(((1 - beta) * base6 + beta * tilt if tilt is not None else base6)[None, :], rel, gamma)[0])
        Wh = np.array([ew, ref_vec, ens_vec, mem[3], mem[15]]); uh = U_batch(Wh, rel, gamma)
        if hed:
            Uh = np.array(hed); sg = float(np.mean(Uh.std(1))) + 1e-9; lam = 4.0 * sg / math.sqrt(len(Uh)); zz = Uh.mean(0) / lam
            qh = np.exp(zz - zz.max()); qh /= qh.sum()
        else:
            qh = np.full(5, 0.2)
        out["Hedge (adapted)"] = float(U_batch((qh @ Wh)[None, :], rel, gamma)[0])
        # ---- MemTrial (frozen memtrial.py) and its variants ----
        idsl = [int(m) for m in ret4]
        for name, val in MT.decide(t, idsl, umem, refavg, ens, gridE).items():
            out[name] = val; out["_open|" + name] = float(MT.log[name][-1][1])
        if t >= warm:
            for k, v in out.items(): res[k].append(v)
        MT.matured(t, idsl, umem, refavg, ens, gridE)
        for m in fm: imp[m] = min(1.0, max(0.0, imp[m] + (0.1 if u_fm > u_ew else -0.1)))
        for m in xp: votes[m] += 1.0 if u_xp > u_ew else -1.0
        refl.append(int(np.argmax(RET[t]))); hed.append(list(uh))
        pastret += [list(REL[t][:, j] / (REL[t][:, j - 1] if j > 0 else np.ones(4)) - 1) for j in range(REL[t].shape[1])]
        for j, m in enumerate(ret4):                             # Banzhaf contribution of each retrieved experience
            on = [umem[mask] for mask in HALF if (mask >> j) & 1]; off = [umem[mask] for mask in HALF if not (mask >> j) & 1]
            obs[m].append(np.mean(on) - np.mean(off))
    means = {k: float(np.mean(v)) for k, v in res.items()}
    return means, means["_open|MemTrial"]


def _job(a):
    gname, n, s = a
    return episode(seed=s, gamma=GAMMA[gname], **REGIMES[n])


if __name__ == "__main__":
    mode, gname, s0, s1 = sys.argv[1], sys.argv[2], int(sys.argv[3]), int(sys.argv[4]); t0 = time.time()
    if mode == "run":
        R = {n: {str(s): episode(seed=s, gamma=GAMMA[gname], **kw) for s in range(s0, s1)} for n, kw in REGIMES.items()}
    elif mode == "prun":
        import multiprocessing as mp
        jobs = [(gname, n, s) for n in REGIMES for s in range(s0, s1)]
        with mp.Pool(4) as pool: outs = pool.map(_job, jobs, chunksize=5)
        R = collections.defaultdict(dict)
        for (g_, n, s), o in zip(jobs, outs): R[n][str(s)] = o
    json.dump(R, open(OUT / f"suite5_{gname}_{s0}_{s1}.json", "w")); print(gname, s0, s1, f"{time.time() - t0:.0f}s")
