"""Print Tables 1, 3 and 4 of the paper from the outputs of reproduce.sh (offline) -> tables/PAPER_TABLES.md.
Utility per decision period in pp per month (bp per day on InvestorBench); mean +- sd over seeds."""
import sys, json, math, pickle
from pathlib import Path
import numpy as np
HERE = Path(__file__).resolve().parent; LAB = HERE.parent
sys.path.insert(0, str(LAB / "memtrial"))
from paper_tables import pb_load, ct_load, ct_matrix, CORE, CT, PB
from memgate import t_sf
EXP = ["FinMem", "MemRL", "Reflexion", "ExpeL"]
ORDER = ["Minimum variance", "Zero-shot (no memory)", "Self-consistency", "Similarity retrieval (top-2)", "Similarity retrieval (top-4)",
         "FinMem", "MemRL", "Reflexion", "ExpeL", "Uplift credit", "Counterfactual selection", "Draft averaging", "Hedge", "MemGate"]
G3 = ("conservative", "balanced", "aggressive")
SC = LAB / "classalloc_and_robustness/sc_runs"


def p_one_sided(d):
    d = np.asarray(d, float); n = len(d)
    if n < 3 or d.std(ddof=1) == 0: return 1.0
    return t_sf(d.mean() / (d.std(ddof=1) / math.sqrt(n)), n - 1)


def column(per_seed, per_unit, trust):
    """per_seed / per_unit: {method: array}; per_unit = per test date (real benchmarks) or per seed (PlantedMem)."""
    c = {k: (float(np.mean(v)), float(np.std(v, ddof=1))) for k, v in per_seed.items()}
    be = max(EXP, key=lambda k: c[k][0]); mt = np.asarray(per_seed["MemGate"]); b = np.asarray(per_seed[be])
    p = p_one_sided(np.asarray(per_unit["MemGate"]) - np.asarray(per_unit[be]))
    c["_improv"] = (100 * (mt.mean() - b.mean()) / b.mean(), 100 * float(np.std(mt - b, ddof=1)) / b.mean(), p, be)
    d = mt - np.asarray(per_seed["1/N"]); c["_minus_1N"] = (float(d.mean()), float(np.std(d, ddof=1))); c["_trust"] = trust
    return c


def portbench(cfg, lab):
    A, dates = pb_load(cfg); per = {k: 100 * A[v].mean((0, 1)) for k, v in PB.items() if v in A}
    unit = {k: 100 * A[v].mean((1, 2)) for k, v in PB.items() if v in A}
    s = json.load(open(SC / "pb_main/SC_EVAL.json"))[lab]["new"]; per["Self-consistency"] = np.array(s["per_seed"])
    R = {p: json.load(open(LAB / "portbench" / f"pb_all_{cfg}_{p}.json")) for p in G3}
    tr = [100 * np.mean([g["open"] for p in G3 for g in R[p]["gate"] if g["world"] == w and g["date"] >= R[p]["split"]]) for w in range(3)]
    return column(per, unit, (float(np.mean(tr)), float(np.std(tr, ddof=1))))


def investorbench():
    T = LAB / "investorbench/turnover"; B = json.load(open(T / "base.json"))["res"]; M = json.load(open(T / "mg_MemGate.json"))
    A = {k: 1e4 * np.array(v, float) for k, v in B.items()}; A["MemGate"] = 1e4 * np.array(M["res"], float)
    per = {k: np.nanmean(v, (0, 1)) for k, v in A.items()}; unit = {k: np.nanmean(v, (1, 2)) for k, v in A.items()}
    per["Self-consistency"] = np.array(json.load(open(SC / "ib_main/SC_EVAL.json"))["new"]["per_seed"])
    tr = 100 * np.nanmean(np.array(M["open"], float), (0, 1))
    return column(per, unit, (float(tr.mean()), float(tr.std(ddof=1))))


def classalloc():
    E = LAB / "classalloc_and_robustness/classalloc/run/eval"
    res = pickle.load(open(E / "base.pkl", "rb"))[0]; mres, _, mop, _ = pickle.load(open(E / "mg0.pkl", "rb"))
    A = {k: 100 * np.asarray(v, float) for k, v in res.items()}; A["MemGate"] = 100 * np.asarray(mres, float)
    per = {k: np.nanmean(v, (0, 1)) for k, v in A.items()}; unit = {k: np.nanmean(v, (1, 2)) for k, v in A.items()}
    per["Self-consistency"] = np.array(json.load(open(SC / "cb_main/SC_EVAL.json"))["new"]["per_seed"])
    tr = 100 * np.asarray(mop, float).mean((0, 1))
    return column(per, unit, (float(tr.mean()), float(tr.std(ddof=1))))


def plantedmem():
    ALL = ct_load(); per = {k: 100 * ct_matrix(ALL, v).mean((1, 2)) for k, v in CT.items()}
    tr = 100 * ct_matrix(ALL, "_open|MemGate").mean((1, 2))
    return column(per, per, (float(tr.mean()), float(tr.std(ddof=1))))


def fmt(m, s, d=2, sign=False): return f"{m:+.{d}f} ± {s:.{d}f}" if sign else f"{m:.{d}f} ± {s:.{d}f}"


if __name__ == "__main__":
    L = ["# Table 1: Spearman correlations (mean ± sd over seeds) and dates with a detectable memory effect", "",
         "| Benchmark | Outcome credit vs. 1/N | Contribution vs. 1/N | Outcome credit vs. contribution | Detectable dates |", "|---|---|---|---|---|"]
    T1 = json.load(open(LAB / "table1/T1C_report.json"))
    for n, r in T1.items():
        a, b, tot = r["detectable_range"]
        L.append(f"| {n} | {fmt(*r['oc_1N'], sign=True)} | {fmt(*r['cf_1N'], sign=True)} | {fmt(*r['oc_cf'], sign=True)} | {a}–{b} / {tot} |")
    cols = [("PortBench-Full", portbench("full-price", "PortBench-Full"), 2), ("PortBench-Raw", portbench("raw-price", "PortBench-Raw"), 2),
            ("InvestorBench (bp/day)", investorbench(), 2), ("ClassAlloc", classalloc(), 3), ("PlantedMem (5 core regimes)", plantedmem(), 2)]
    L += ["", "# Table 3: main results", "", "| Method | " + " | ".join(c for c, _, _ in cols) + " |", "|---" * (len(cols) + 1) + "|"]
    for m in ORDER:
        L.append(f"| {'MemTrial (ours)' if m == 'MemGate' else m} | " + " | ".join(fmt(*c[m], d=d) for _, c, d in cols) + " |")
    L.append("| Improv. (%) | " + " | ".join(f"{c['_improv'][0]:+.1f} ± {c['_improv'][1]:.1f}{'*' if c['_improv'][2] < 0.05 and c['_improv'][0] > 0 else ''}"
                                         f" (vs {c['_improv'][3]}; p = {c['_improv'][2]:.3f})" for _, c, _ in cols) + " |")
    L.append("| MemTrial − 1/N | " + " | ".join(fmt(*c["_minus_1N"], d=d, sign=True) for _, c, d in cols) + " |")
    L.append("| Trust rate (%) | " + " | ".join(fmt(*c["_trust"], d=1) for _, c, _ in cols) + " |")
    L.append("| 1/N | " + " | ".join(fmt(*c["1/N"], d=d) for _, c, d in cols) + " |")
    M = json.load(open(LAB / "ablation/NT_MATRIX2.json"))["rows"]
    C4 = ["PortBench-Full", "PortBench-Raw", "InvestorBench", "ClassAlloc", "PlantedMem no signal", "PlantedMem signal", "PlantedMem informative"]
    L += ["", "# Figure 4 (and Table D.4): ablation, utility of MemTrial and change in % (mean ± sd over seeds; =: same actions)", "",
          "| Variant | " + " | ".join(C4) + " |", "|---" * (len(C4) + 1) + "|"]
    for v, r in M.items():
        cells = [fmt(r[c]["mean"], r[c]["sd"], d=3 if c == "ClassAlloc" else 2) if v == "MemTrial" else
                 ("=" if r[c]["identical"] else fmt(r[c]["delta_pct"], r[c]["delta_sd"], d=2 if f"{abs(r[c]['delta_pct']):.1f}" == "0.0" else 1, sign=True))
                 for c in C4]
        L.append(f"| {v} | " + " | ".join(cells) + " |")
    (HERE / "PAPER_TABLES.md").write_text("\n".join(L) + "\n"); print("\n".join(L))
