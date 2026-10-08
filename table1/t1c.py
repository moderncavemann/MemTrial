"""Compact Table 1 (one row per benchmark). Analysis only: reads published caches and result files, writes T1C_* files here.
Per seed and investor, the same quantities as tables/sd_{pb,ib,cb}.py (Table 1): outcome credit oc of experience j on a date
(mean utility of the drafts whose subset contains j, over the 16 subsets), counterfactual contribution cf (mean with j minus mean
without j) and the utility of 1/N on that date. New: Spearman(oc, cf). Each correlation is averaged over the three investors per seed;
tables report mean +- sd over the three seeds. Gaps to 1/N: per seed, mean over test dates and investors (pp per month; bp per day
on InvestorBench); the best experience-learning agent has the highest mean over investors and seeds, as in the main table.
Detectable dates: the published per-investor F-test counts (MONTHLY_DIAGNOSTICS.md, IB_DIAG_turnover.json, classalloc RESULTS.json).
usage: python3 t1c.py pb | ib | cb | report"""
import sys, json, pickle, re
from pathlib import Path
import numpy as np
HERE = Path(__file__).resolve().parent; LAB = HERE.parent
sys.path.insert(0, str(LAB / "tables"))
from sd_common import ms, spearman, G3          # sd_common also puts memtrial/, portbench/, investorbench/, classalloc_and_robustness/ on the path
EXP = ["FinMem", "MemRL", "Reflexion", "ExpeL"]
STAGE = sys.argv[1]


def dump(name, obj):
    (HERE / f"T1C_{name}.json").write_text(json.dumps(obj, indent=1)); print(f"wrote T1C_{name}.json")


def corr3(oc, cf, mk):
    n = len(oc)
    return {"oc_1N": [spearman(oc[i], mk) for i in range(n)], "cf_1N": [spearman(cf[i], mk) for i in range(n)],
            "oc_cf": [spearman(oc[i], cf[i]) for i in range(n)]}


if STAGE == "pb":
    import monthly_lib as ML
    MON = LAB / "portbench"; EXPK = {"FinMem": "FinMem", "MemRL": "MemRL", "Reflexion": "Reflexion", "ExpeL": "ExpeL (adapted)"}
    rows, ids, dates, split = ML.load_monthly(); D = ML.Data(rows, ids, dates, split, cache=ML.load_cache())
    out = {}
    for cfg in ("full-price", "raw-price"):
        for p in G3:
            key = (cfg, p); kind = f"U_{p}"; S = D.score[key]; M16 = [str(m) for m in range(16)]
            ds = [d for d in D.dates if d in S and all(all(m in S[d][r] for m in M16) for r in (0, 1, 2) if r in S[d]) and all(r in S[d] for r in (0, 1, 2))]
            mk = []; oc = [[], [], []]; cf = [[], [], []]
            for d in ds:
                u1n = float(D.U(D.ref[(d, p, "1/N")], d, p)[0])
                for j in range(4):
                    on = [m for m in M16 if (int(m) >> j) & 1]; off = [m for m in M16 if not (int(m) >> j) & 1]
                    for r in range(3):
                        a = np.mean([S[d][r][m][kind] for m in on]); b = np.mean([S[d][r][m][kind] for m in off])
                        oc[r].append(a); cf[r].append(a - b)
                    mk.append(u1n)
            R = json.load(open(MON / f"pb_all_{cfg}_{p}.json")); te = sorted(d for d in R["res"]["1/N"] if d >= R["split"])
            per = {k: 100 * np.array([R["res"][k][d] for d in te]).mean(0) for k in R["res"]}
            util = {"1/N": per["1/N"].tolist(), "Counterfactual selection": per["Counterfactual selection (no gate)"].tolist()}
            util.update({k: per[v].tolist() for k, v in EXPK.items()})
            if "MemGate" in per: util["MemGate"] = per["MemGate"].tolist()
            out[f"{cfg}|{p}"] = {"n_dates": len(ds), "n_test": len(te), **corr3(oc, cf, mk), "util": util}
            print(cfg, p, len(ds), {k: np.round(v, 4).tolist() for k, v in corr3(oc, cf, mk).items()}, flush=True)
    dump("pb", out)

if STAGE == "ib":
    import ib_turnover as IT
    C = IT.load(); W, seeds, dates, ret = C["W"], C["seeds"], C["dates"], C["ret"]
    J = json.load(open(LAB / "investorbench/ib_all_turnover.json")); RR = {k: np.array(v, float) for k, v in J["res"].items()}
    out = {}
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
        util = {k: (1e4 * np.nanmean(RR[k][:, pi, :], 0)).tolist() for k in EXP + ["1/N", "Counterfactual selection", "MemGate"] if k in RR}
        out[p] = {"n_dates": len(dates), **corr3(oc, cf, mk), "util": util}
        print(p, len(dates), {k: np.round(v, 4).tolist() for k, v in corr3(oc, cf, mk).items()}, flush=True)
    dump("ib", out)

if STAGE == "cb":
    import cb_eval as CE
    EV = LAB / "classalloc_and_robustness/classalloc/run/eval"; C = pickle.load(open(EV / "cache.pkl", "rb"))
    res, cost, held = pickle.load(open(EV / "base.pkl", "rb")); mres = np.asarray(pickle.load(open(EV / "mg0.pkl", "rb"))[0], float)
    W, seeds, dates, rel = C["W"], C["seeds"], C["dates"], C["rel"]; out = {}; nxt = CE.to_next(dates)
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
        util = {k: (100 * np.nanmean(np.asarray(res[k], float)[:, pi, :], 0)).tolist() for k in EXP + ["1/N", "Counterfactual selection"]}
        util["MemGate"] = (100 * np.nanmean(mres[:, pi, :], 0)).tolist()
        out[p] = {"n_dates": len(dates), **corr3(oc, cf, mk), "util": util}
        print(p, len(dates), {k: np.round(v, 4).tolist() for k, v in corr3(oc, cf, mk).items()}, flush=True)
    dump("cb", out)

if STAGE == "report":
    det_pb = {k: (v["detectable_dates"], v["dates"]) for k, v in json.load(open(LAB / "portbench/DETECTABILITY.json")).items()}   # portbench/detectability.py
    IBD = json.load(open(LAB / "investorbench/IB_DIAG_turnover.json"))
    CBD = json.load(open(LAB / "classalloc_and_robustness/classalloc/run/eval/RESULTS.json"))["table1"]
    T = {s: json.load(open(HERE / f"T1C_{s}.json")) for s in ("pb", "ib", "cb")}
    BENCH = {"PortBench-Full": ("pb", "full-price|"), "PortBench-Raw": ("pb", "raw-price|"), "InvestorBench": ("ib", ""), "ClassAlloc": ("cb", "")}
    rep = {}
    for name, (s, pre) in BENCH.items():
        inv = [T[s][pre + p] for p in G3]; row = {}
        for c in ("oc_1N", "cf_1N", "oc_cf"):
            row[c] = ms(np.mean([v[c] for v in inv], 0))                       # mean over investors per seed, then mean +- sd over seeds
            row[c + "_by_investor"] = {p: ms(v[c]) for p, v in zip(G3, inv)}
        U = {k: np.mean([v["util"][k] for v in inv], 0) for k in inv[0]["util"]}   # per seed, mean over investors
        best = max(EXP, key=lambda k: U[k].mean())
        row.update(best_agent=best, best_minus_1N=ms(U[best] - U["1/N"]), cfsel_minus_1N=ms(U["Counterfactual selection"] - U["1/N"]),
                   every_exp_minus_1N={k: ms(U[k] - U["1/N"]) for k in EXP}, one_over_N=ms(U["1/N"]),
                   util={k: ms(v) for k, v in U.items()})
        if "MemGate" in U: row["memtrial_minus_1N"] = ms(U["MemGate"] - U["1/N"])
        if s == "pb": dd = [det_pb[pre + p] for p in G3]
        elif s == "ib": dd = [(IBD[p]["detectable_days"], IBD[p]["days"]) for p in G3]
        else: dd = [(CBD[p]["detectable_months"], CBD[p]["months"]) for p in G3]
        row["detectable_by_investor"] = dict(zip(G3, dd)); row["detectable_range"] = [min(a for a, _ in dd), max(a for a, _ in dd), dd[0][1]]
        row["detectable_pct_mean"] = 100 * float(np.mean([a / b for a, b in dd]))
        rep[name] = row
    dump("report", rep)
    f = lambda x: f"{x[0]:+.2f} ± {x[1]:.2f}"
    L = ["| Benchmark | oc vs 1/N | cf vs 1/N | oc vs cf | detectable | best EL − 1/N | CF sel − 1/N |", "|---|---|---|---|---|---|---|"]
    for n, r in rep.items():
        a, b, tot = r["detectable_range"]
        L.append(f"| {n} | {f(r['oc_1N'])} | {f(r['cf_1N'])} | {f(r['oc_cf'])} | {a}–{b} / {tot} ({r['detectable_pct_mean']:.0f}%) | "
                 f"{r['best_agent']} {f(r['best_minus_1N'])} | {f(r['cfsel_minus_1N'])} |")
    (HERE / "T1C.md").write_text("\n".join(L) + "\n"); print("\n".join(L))
    for n, r in rep.items(): print(n, "every EL agent - 1/N:", {k: [round(v[0], 3), round(v[1], 3)] for k, v in r["every_exp_minus_1N"].items()}, "1/N", r["one_over_N"], "MemTrial-1/N", r.get("memtrial_minus_1N"))
