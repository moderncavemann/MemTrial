"""PortBench test months one at a time (Appendix D.6; offline, no API calls; python3 pb_months.py, after portbench/pb_all.py
and portbench/monthly_lib.py) -> PB_MONTHS.json.

For each configuration: the utility of every method of Table 3 (pp per month, mean over test months, investors and seeds)
with all test months and with each test month left out, and the best LLM-based method in each case. Self-consistency
averages each seed's own three memory-free draws, as in classalloc_and_robustness/sc_eval.py.
For the last test month (decision of 2024-11-01): the return of the equal-weighted cryptocurrency class over its holding
window, the cryptocurrency weight and gross return of 1/N for each investor, and the cryptocurrency weight of the drafts of
FinMem, MemRL and Reflexion (balanced investor, PortBench-Full)."""
import sys, json
from pathlib import Path
import numpy as np
HERE = Path(__file__).resolve().parent; LAB = HERE.parent
sys.path.insert(0, str(LAB / "portbench")); sys.path.insert(0, str(LAB))
import monthly_lib as ML
G3 = ("conservative", "balanced", "aggressive"); CFGS = {"full-price": "PortBench-Full", "raw-price": "PortBench-Raw"}
KEYS = {"1/N": "1/N", "Minimum variance": "Minimum variance", "Zero-shot": "Zero-shot (no memory)",
        "Similarity retrieval (top-2)": "Similarity retrieval (top-2)", "Similarity retrieval (top-4)": "Similarity retrieval (top-4)",
        "FinMem": "FinMem", "MemRL": "MemRL", "Reflexion": "Reflexion", "ExpeL": "ExpeL (adapted)",
        "Uplift credit": "Uplift credit (UpliftMem-style)", "Counterfactual selection": "Counterfactual selection (no gate)",
        "Draft averaging": "Draft averaging (8 drafts)", "Hedge": "Hedge (memory families)", "MemTrial": "MemTrial"}
RULE = ("1/N", "Minimum variance")
LAST = "2024-11-01"

rows, ids, dates, split = ML.load_monthly()
D = ML.Data(rows + ML.load_rows("sc"), ids, dates, split, cache=ML.load_cache())    # with the extra memory-free draws
out = {"split": split, "last_month": LAST}
for cfg, col in CFGS.items():
    R = {p: json.load(open(LAB / "portbench" / f"pb_all_{cfg}_{p}.json"))["res"] for p in G3}
    test = sorted(d for d in R["balanced"]["1/N"] if d >= split)
    A = {k: 100 * np.array([[R[p][v][d] for p in G3] for d in test], float) for k, v in KEYS.items()}   # months x investors x seeds
    sc = np.full((len(test), 3, 3), np.nan)
    for di, d in enumerate(test):
        for pi, p in enumerate(G3):
            W = D.proj[(cfg, p)][d]
            for s in range(3):
                vs = [W[r]["0"] for r in (s, 3 + 2 * s, 4 + 2 * s) if "0" in W.get(r, {})]
                sc[di, pi, s] = 100 * float(D.U(np.mean(vs, 0), d, p)[0])
    A["Self-consistency"] = sc
    def table(keep):
        u = {k: float(a[keep].mean()) for k, a in A.items()}
        llm = {k: v for k, v in u.items() if k not in RULE}
        return {"utility": u, "best_llm_based": max(llm, key=llm.get)}
    allm = np.ones(len(test), bool); res = {"months": test, "all": table(allm), "without": {}}
    for i, d in enumerate(test):
        keep = allm.copy(); keep[i] = False; res["without"][d] = table(keep)
    res["months_whose_removal_changes_the_best_llm_based"] = [d for d, t in res["without"].items() if t["best_llm_based"] != res["all"]["best_llm_based"]]
    out[col] = res
    t = res["without"][LAST]["utility"]
    print(f"{col}: best LLM-based {res['all']['best_llm_based']}; removal changes it for {res['months_whose_removal_changes_the_best_llm_based']}")
    print("   without the last month:", {k: round(v, 3) for k, v in sorted(t.items(), key=lambda kv: -kv[1])[:5]})
# the last test month
RS = ML.RS; uni = D.uni[LAST]; crypto = [i for i, a in enumerate(uni) if str(RS.cls(a)).lower().startswith("crypto")]
R1 = D.rel[LAST][:, -1] - 1
last = {"crypto_assets": [uni[i] for i in crypto], "crypto_ew_return_pct": 100 * float(np.mean(R1[crypto])), "one_over_n": {}, "agents_crypto_weight": {}}
for p in G3:
    x = D.ref[(LAST, p, "1/N")]; last["one_over_n"][p] = {"crypto_weight": float(x[crypto].sum()), "gross_return_pct": 100 * float(x @ R1)}
for arm in ("FinMem", "MemRL", "Reflexion"):
    W = D.proj[("full-price", "balanced")][LAST]
    last["agents_crypto_weight"][arm] = [float(W[r][arm][crypto].sum()) for r in sorted(W) if arm in W[r]]
out["last_month_market"] = last
(HERE / "PB_MONTHS.json").write_text(json.dumps(out, indent=1))
print(f"last month: equal-weighted cryptocurrency class {last['crypto_ew_return_pct']:+.1f}%; 1/N (balanced) holds "
      f"{100 * last['one_over_n']['balanced']['crypto_weight']:.0f}% of it and gains {last['one_over_n']['balanced']['gross_return_pct']:.1f}%; "
      f"agents' drafts hold {100 * min(min(v) for v in last['agents_crypto_weight'].values()):.0f}-"
      f"{100 * max(max(v) for v in last['agents_crypto_weight'].values()):.0f}%")
