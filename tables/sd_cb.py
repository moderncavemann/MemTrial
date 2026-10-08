"""ClassAlloc: sd over the three seeds for every ClassAlloc number reported without one (python3 sd_cb.py -> SD_cb.json).
Per seed = mean over the 71 test months and the three investors, pp per month. Reads classalloc/run/eval/{cache,base,mt0}.pkl;
the ablation variants are re-run with the loop of cb_eval_extra.run_variant (frozen memtrial.py), keeping per-seed values."""
import json, pickle
import numpy as np
from sd_common import *
import memtrial as MT
from memtrial import MemTrialBank, GRID, ftrl_q
import cb_eval as CE, cb_eval_extra as CX
EV = LAB / "classalloc_and_robustness/classalloc/run/eval"; C = pickle.load(open(EV / "cache.pkl", "rb"))
res, cost, held = pickle.load(open(EV / "base.pkl", "rb")); mres, mcost, mop, _ = pickle.load(open(EV / "mt0.pkl", "rb"))
ps = lambda a: 100 * np.nanmean(np.asarray(a, float), (0, 1))
R = {k: ps(v) for k, v in res.items()}; R["MemTrial"] = ps(mres)
EXP = ["FinMem", "MemRL", "Reflexion", "ExpeL"]; be = max(EXP, key=lambda k: R[k].mean())
out = {"main": {"best_experience_agent": be, "MemTrial": ms(R["MemTrial"]), "improv": ms(R["MemTrial"] - R[be]),
                "minus_1N": ms(R["MemTrial"] - R["1/N"]), "trust_pct": ms(100 * mop.mean((0, 1)))}}


def run_variant(name):
    cfg = CX.VARIANTS[name]; W, top4, seeds, dates, rel, Z = C["W"], C["top4"], C["seeds"], C["dates"], C["rel"], C["Z"]
    rs = np.full((len(dates), 3, len(seeds)), np.nan); opened = np.zeros_like(rs); cl = cfg.get("closed", "ftrl"); nxt = CE.to_next(dates)
    for pi, (p, (M, mfl, gam)) in enumerate(CE.INV.items()):
        ew = CE.project(np.full(CE.K, 1.0 / CE.K), M, mfl)
        for si, s in enumerate(seeds):
            bank = MemTrialBank(Z, {name: cfg}, masks=CE.HALF); h = ew.copy(); fb = CE.Feedback()
            for di, t in enumerate(dates):
                for a_ in fb.ready(t): bank.matured(*a_)
                w = {k: CE.project(v, M, mfl) for k, v in W[(t, s)].items()}; Rl = rel[t]; ids = top4[t]
                ens = np.mean([w[f"m{m}"] for m in CE.HALF], 0)
                U = {m: float(CE.util(w[f"m{m}"], Rl, gam, h)[0][0]) for m in CE.HALF}
                u_ref = float(CE.util(ew, Rl, gam, h)[0][0]); u_ens = float(CE.util(ens, Rl, gam, h)[0][0])
                grid = CE.util(GRID[:, None] * ew[None, :] + (1 - GRID)[:, None] * ens[None, :], Rl, gam, h)[0]
                q = ftrl_q(bank.sumG, bank.n, bank.sig, 0.5 if cl == "uniform" else 0.9, 4.0)
                o = bank.decide(t, ids, U, u_ref, u_ens, grid)[name]; op = bool(bank.log[name][-1][1])
                if op:
                    lm = cfg.get("learner", "content")
                    if lm == "auto":
                        sc = {m: np.mean(bank.G[m].scores) if len(bank.G[m].scores) >= MT.MIN_SCORES else -np.inf for m in ("identity", "content")}
                        lm = "content" if sc["content"] > sc["identity"] else "identity"
                    v = [bank.L[lm].predict(hh) for hh in ids]
                    best = max(CE.HALF, key=lambda k: (sum(v[j] for j in range(len(ids)) if (k >> j) & 1), -bin(k).count("1"))); X = w[f"m{best}"]
                elif cl == "ens": X = ens
                elif cl == "ref": X = ew
                else: X = q * ew + (1 - q) * ens
                if abs(float(CE.util(X, Rl, gam, h)[0][0]) - o) > 1e-6: raise RuntimeError("mismatch")
                fb.add(t, (t, ids, U, u_ref, u_ens, grid)); rs[di, pi, si] = o; opened[di, pi, si] = op; h = CE.drift(X, nxt[t])
    return rs, opened


EXTRA = json.load(open(EV / "EXTRA.json"))["ablation"]; out["variants"] = {}
for n in CX.VARIANTS:
    rs, op = run_variant(n); u = ms(ps(rs)); tr = ms(100 * op.mean((0, 1)))
    assert abs(u[0] - EXTRA[n]["mean"]) < 1e-9 and abs(tr[0] - EXTRA[n]["trust_pct"]) < 1e-9, (n, u, EXTRA[n])
    out["variants"][n] = {"util": u, "trust_pct": tr}; print(n, out["variants"][n], flush=True)
# ---- Table 1, per seed
W, seeds, dates, rel = C["W"], C["seeds"], C["dates"], C["rel"]; t1 = {}; nxt = CE.to_next(dates)
for pi, (p, (M, mfl, gam)) in enumerate(CE.INV.items()):
    ew = CE.project(np.full(CE.K, 1.0 / CE.K), M, mfl); h = ew.copy(); oc = [[] for _ in seeds]; cf = [[] for _ in seeds]; mk = []
    for t in dates:
        U = np.array([[float(CE.util(CE.project(W[(t, s)][f"m{m}"], M, mfl), rel[t], gam, h)[0][0]) for m in range(16)] for s in seeds])
        u1n = float(CE.util(ew, rel[t], gam, h)[0][0])
        for j in range(4):
            on = [m for m in range(16) if (m >> j) & 1]; off = [m for m in range(16) if not (m >> j) & 1]
            for si in range(len(seeds)): oc[si].append(U[si, on].mean()); cf[si].append(U[si, on].mean() - U[si, off].mean())
            mk.append(u1n)
        h = CE.drift(ew, nxt[t])
    ex = {k: np.nanmean(res[k][:, pi, :]) for k in EXP}; best = max(ex, key=ex.get)
    d1 = 100 * (np.nanmean(res[best][:, pi, :], 0) - np.nanmean(res["1/N"][:, pi, :], 0))
    d2 = 100 * (np.nanmean(res["Counterfactual selection"][:, pi, :], 0) - np.nanmean(res["1/N"][:, pi, :], 0))
    t1[p] = {"outcome_vs_1N": ms([spearman(oc[i], mk) for i in range(len(seeds))]), "cf_vs_1N": ms([spearman(cf[i], mk) for i in range(len(seeds))]),
             "best_agent": best, "best_minus_1N": ms(d1), "cfsel_minus_1N": ms(d2)}
out["table1"] = t1
save("cb", out); print(json.dumps(out["main"])); print(json.dumps(t1))
