"""InvestorBench: sd over the three seeds for every InvestorBench number reported without one (python3 sd_ib.py -> SD_ib.json).
Per seed = mean over the 149 test days and the three investors, in bp per day (trust rates in %). Published files only
(turnover/*.json, ib_all_turnover.json, ib_runs/<v>/eval/*.json); Table 1 credits are recomputed per seed with the
unchanged ib_turnover functions (the published Table 1 averages the subset utilities over the three seeds first)."""
import json, re
import numpy as np
from sd_common import *
import ib_turnover as IT
TO = LAB / "investorbench/turnover"; SR = LAB / "classalloc_and_robustness/ib_runs"
GAM = [IT.INV[p][2] for p in G3]
EXP = ["FinMem", "MemRL", "Reflexion", "ExpeL"]
LLMB = ["Zero-shot (no memory)", "Self-consistency", "Similarity retrieval (top-2)", "Similarity retrieval (top-4)", "FinMem", "MemRL", "Reflexion",
        "ExpeL", "Uplift credit", "Counterfactual selection", "Draft averaging", "Hedge"]
ps = lambda a: 1e4 * np.nanmean(np.asarray(a, float), (0, 1))          # per seed, bp/day
pt = lambda a: 100 * np.nanmean(np.asarray(a, float), (0, 1))          # per seed, %
mg = lambda folder, slug: json.load(open(folder / f"mg_{slug}.json"))
out = {}
# ---- main table and robustness rows, every run
RUNS = {"gpt-4.1-mini, 0.7 (main)": TO, "gpt-4.1-mini, 0": SR / "t0.0/eval", "gpt-4.1-mini, 0.3": SR / "t0.3/eval", "gpt-4.1-mini, 1.0": SR / "t1.0/eval",
        "gpt-4.1-nano, 0.7": SR / "nano/eval", "gpt-5-mini": SR / "gpt5mini/eval", "Llama-3.3-70B, 0.7": SR / "llama70b/eval"}
out["runs"] = {}
for lab, f in RUNS.items():
    if not (f / "base.json").exists(): print(f"{lab}: not evaluated here (needs the logs of that run)"); continue
    B = json.load(open(f / "base.json")); R = {k: ps(v) for k, v in B["res"].items()}
    m = mg(f, "MemGate"); R["MemGate"] = ps(m["res"]); R["MemGate | no gate"] = ps(mg(f, "MemGate_no_gate")["res"])
    be = max(EXP, key=lambda k: R[k].mean()); bo = max(LLMB, key=lambda k: R[k].mean())
    out["runs"][lab] = {"rows": {k: ms(v) for k, v in R.items()}, "best_experience_agent": be, "best_other": bo,
                        "best_other_ms": ms(R[bo]), "improv": ms(R["MemGate"] - R[be]), "minus_1N": ms(R["MemGate"] - R["1/N"]),
                        "trust_pct": ms(pt(m["open"]))}
    print(f"{lab:26s} MemTrial {R['MemGate'].mean():.2f}±{R['MemGate'].std(ddof=1):.2f} improv {out['runs'][lab]['improv']} -1/N {out['runs'][lab]['minus_1N']} trust {out['runs'][lab]['trust_pct']} best other {bo} {out['runs'][lab]['best_other_ms']}")
# ---- ablation, trust test sensitivity and anchor (main run)
ABL = {"MemGate": "MemGate", "per-experience only": "MemGate_identity_learner", "content-aware only": "MemGate_content_learner_only",
       "in-sample F-test": "MemGate_F_test_gate", "no trust test": "MemGate_no_gate", "fall back to the draft average": "MemGate_closed_ensemble",
       "fall back to the reference": "MemGate_closed_reference", "uniform prior": "MemGate_uniform_prior",
       "alpha=0.01": "MemGate_gate_0_01", "alpha=0.02": "S_alpha_0_02", "alpha=0.1": "S_alpha_0_1", "alpha=0.2": "MemGate_gate_0_2",
       "minscores=5": "S_minscores_5", "minscores=20": "S_minscores_20"}
out["variants"] = {}
for lab, slug in ABL.items():
    J = mg(TO, slug); out["variants"][lab] = {"util": ms(ps(J["res"])), "trust_pct": ms(pt(J["open"]))}
out["anchor"] = {}
for a0 in (0.5, 0.75, 0.9, 0.95):
    for l0 in (1.0, 4.0, 16.0):
        slug = "MemGate" if (a0, l0) == (0.9, 4.0) else "S_" + re.sub(r"[^A-Za-z0-9]+", "_", f"a0={a0},lam0={l0}").strip("_")
        out["anchor"][f"{a0},{l0}"] = ms(ps(mg(TO, slug)["res"]))
# ---- decomposition (supplement): utility, gross return, trading cost, risk penalty, per seed
B = json.load(open(TO / "base.json")); dec = {}
for k in ["1/N", "Zero-shot (no memory)", "Similarity retrieval (top-4)", "Draft averaging", "ExpeL"]:
    u, g, c = (np.array(B[x][k], float) for x in ("res", "gross", "cost"))
    dec[k] = {"utility": ms(ps(u)), "gross": ms(ps(g)), "cost": ms(ps(c)), "risk": ms(ps(g - c - u))}
m = mg(TO, "MemGate"); u = np.array(m["res"], float); c = np.array(m["cost"], float)
net = np.stack([(1 - np.sqrt(1 - 2 * GAM[i] * u[:, i])) / GAM[i] for i in range(3)], 1); g = net + c
dec["MemTrial"] = {"utility": ms(ps(u)), "gross": ms(ps(g)), "cost": ms(ps(c)), "risk": ms(ps(g - c - u))}
out["decomposition"] = dec
# ---- Table 1, per seed
C = IT.load(); W, seeds, dates, ret = C["W"], C["seeds"], C["dates"], C["ret"]
J = json.load(open(LAB / "investorbench/ib_all_turnover.json")); RR = {k: np.array(v, float) for k, v in J["res"].items()}
t1 = {}
for pi, (p, (M, mfl, gam)) in enumerate(IT.INV.items()):
    ew = IT.project(np.full(5, 0.2), M, mfl); h = ew.copy(); oc = [[] for _ in seeds]; cf = [[] for _ in seeds]; mk = []
    for t in dates:
        r = ret[t]
        U = np.array([[float(IT.util(IT.project(W[(t, s)][f"m{m_}"], M, mfl), r, gam, h)[0]) for m_ in range(16)] for s in seeds])
        u1n = float(IT.util(ew, r, gam, h)[0])
        for j in range(4):
            on = [m_ for m_ in range(16) if (m_ >> j) & 1]; off = [m_ for m_ in range(16) if not (m_ >> j) & 1]
            for si in range(len(seeds)): oc[si].append(U[si, on].mean()); cf[si].append(U[si, on].mean() - U[si, off].mean())
            mk.append(u1n)
        h = IT.drift(ew, r)
    ex = {k: np.nanmean(RR[k][:, pi, :]) for k in EXP}; best = max(ex, key=ex.get)
    d1 = 1e4 * (np.nanmean(RR[best][:, pi, :], 0) - np.nanmean(RR["1/N"][:, pi, :], 0))
    d2 = 1e4 * (np.nanmean(RR["Counterfactual selection"][:, pi, :], 0) - np.nanmean(RR["1/N"][:, pi, :], 0))
    t1[p] = {"outcome_vs_1N": ms([spearman(oc[i], mk) for i in range(len(seeds))]), "cf_vs_1N": ms([spearman(cf[i], mk) for i in range(len(seeds))]),
             "best_agent": best, "best_minus_1N": ms(d1), "cfsel_minus_1N": ms(d2)}
out["table1"] = t1
save("ib", out)
print(json.dumps(t1, indent=0)[:1500]); print(json.dumps(out["decomposition"])[:800])
