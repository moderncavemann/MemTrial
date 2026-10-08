"""Return and risk from the wealth paths (offline; python3 return_risk.py, after wealth_pb.py, wealth_ib.py, wealth_cb.py and
market.py) -> RETURN_RISK.json.

table4: the balanced investor with gpt-4.1-mini (Table 4): cumulative net return CR (%), annualized Sharpe ratio SR and
maximum drawdown MDD (%) of every method, mean and sd over the three seeds, for PortBench-Full, InvestorBench and ClassAlloc.
by_benchmark: the same and the wealth relative to 1/N (end of the test period, falling and rising market phases) and the
survival against 1/N, each seed averaging every available setting of a benchmark (Tables D.6-D.8; the paper's tables also
include the runs with other LLMs, whose logs are not included).
Wealth paths: on InvestorBench and ClassAlloc the value of the account from each decision date to the next; on PortBench,
whose decisions each start from the equal-weight book, the compounded net returns of the decisions."""
import json, glob, os
import numpy as np
os.chdir(os.path.dirname(os.path.abspath(__file__)))
PER_YEAR = {"PortBench-Full": 12, "PortBench-Raw": 12, "PortBench": 12, "InvestorBench": 252, "ClassAlloc": 12}
INV = ["conservative", "balanced", "aggressive"]
REN = {"MemGate": "MemTrial", "ExpeL (adapted)": "ExpeL", "Self-consistency (3 drafts)": "Self-consistency", "Uplift credit (UpliftMem-style)": "Uplift credit",
       "Counterfactual selection (no gate)": "Counterfactual selection", "Draft averaging (8 drafts)": "Draft averaging", "Hedge (memory families)": "Hedge"}
ORDER = ["Zero-shot (no memory)", "FinMem", "MemRL", "Reflexion", "ExpeL", "MemTrial"]
MK = json.load(open("MARKET.json"))


def path(a): return np.vstack([np.ones((1, a.shape[1])), np.cumprod(1 + a, 0)])


def phases(rets, thr=0.05):
    """falling phases: from a peak of the equity index to the following trough, when the index lost at least thr."""
    W = np.concatenate([[1.0], np.cumprod(1 + np.array(rets))]); out = []; n = len(W); peak = 0; j = 1
    while j < n:
        if W[j] >= W[peak]: peak = j; j += 1; continue
        if W[j] <= W[peak] * (1 - thr):
            k = j; tr = j
            while k < n and W[k] < W[peak]:
                if W[k] < W[tr]: tr = k
                k += 1
            out.append((peak, tr)); peak = tr; j = tr + 1; continue
        j += 1
    return out


def pb():
    out = {}
    for cfg in ("full-price", "raw-price"):
        for p in INV:
            J = json.load(open(f"WEALTH_PB_{cfg}_{p}.json")); test = [d for d in J["dates"] if d >= J["split"]]; net = {}
            for k, byd in J["net"].items():
                if k.startswith("MemGate |"): continue
                if all(d in byd for d in test): net[REN.get(k, k)] = np.array([byd[d] for d in test], float)
            out[(cfg, p)] = {"dates": test, "net": net}
    return out


def ib(run=None):
    J = json.load(open("WEALTH_IB.json")); run = run or list(J["runs"])[0]
    return {p: {"dates": J["dates"], "net": {REN.get(k, k): np.array(v[p]["net_by_seed"], float) for k, v in J["runs"][run].items()}} for p in J["investors"]}


def cb(fname="WEALTH_CB.json"):
    J = json.load(open(fname)); out = {}
    for pi, p in enumerate(J["investors"]):
        out[p] = {"dates": J["dates"], "net": {REN.get(k, k): np.array(v, float)[:, pi, :] for k, v in J["net"].items()}}
    return out


def ms(x): x = np.asarray(x, float); return [float(x.mean()), float(x.std(ddof=1))]


def risk(x, bench):
    W = path(x); return 100 * (W[-1] - 1), x.mean(0) / x.std(0, ddof=1) * np.sqrt(PER_YEAR[bench]), 100 * (1 - W / np.maximum.accumulate(W, 0)).max(0)


out = {"table4": {}, "by_benchmark": {}}
P_, I_, C_ = pb(), ib(), cb()
for bench, v in (("PortBench-Full", P_[("full-price", "balanced")]), ("InvestorBench", I_["balanced"]), ("ClassAlloc", C_["balanced"])):
    out["table4"][bench] = {}
    for k, x in v["net"].items():
        cr, sr, mdd = risk(x, bench); out["table4"][bench][k] = {"cr": ms(cr), "sr": ms(sr), "mdd": ms(mdd)}
S = [("PortBench", v) for v in P_.values()] + [("InvestorBench", v) for v in I_.values()]
for f in sorted(glob.glob("WEALTH_CB*.json")): S += [("ClassAlloc", v) for v in cb(f).values()]
for bench in ("PortBench", "InvestorBench", "ClassAlloc"):
    res = {}
    for k in ["1/N"] + ORDER:
        term, fall, rise, cr, sr, mdd = [], [], [], [], [], []
        for b, v in S:
            if b != bench: continue
            R = path(v["net"]["1/N"]); W = path(v["net"][k]); term.append(100 * (W[-1] / R[-1] - 1))
            mk = dict(zip(MK[bench]["dates"], MK[bench]["equity"])); m = np.zeros(len(v["dates"]), bool)
            for a, e in phases([mk[t] for t in v["dates"]]): m[a:e] = True
            d = np.log1p(v["net"][k]) - np.log1p(v["net"]["1/N"])
            fall.append(100 * (np.exp(d[m].sum(0)) - 1)); rise.append(100 * (np.exp(d[~m].sum(0)) - 1))
            c_, s_, m_ = risk(v["net"][k], bench); cr.append(c_); sr.append(s_); mdd.append(m_)
        f2 = lambda L: ms(np.mean(L, 0))
        res[k] = {"end_vs_1N": f2(term), "falling_vs_1N": f2(fall), "rising_vs_1N": f2(rise), "cr": f2(cr), "sr": f2(sr), "mdd": f2(mdd)}
    out["by_benchmark"][bench] = res
surv = {}
for k in ORDER:
    short = np.array([np.max(1 - path(v["net"][k]) / path(v["net"]["1/N"]), 0) for b, v in S])
    surv[k] = {f"shortfall<{t}%": ms(100 * (short < t / 100).mean(0)) for t in (2, 5, 10)}
    surv[k]["median_largest_shortfall"] = ms(100 * np.median(short, 0))
out["survival"] = surv; out["runs_per_seed"] = len(S)
json.dump(out, open("RETURN_RISK.json", "w"), indent=1)
for bench, rows in out["table4"].items():
    print(bench)
    for k, r in rows.items(): print(f"   {k:28s} CR {r['cr'][0]:6.1f}±{r['cr'][1]:.1f}  SR {r['sr'][0]:.2f}±{r['sr'][1]:.2f}  MDD {r['mdd'][0]:5.1f}±{r['mdd'][1]:.1f}")
