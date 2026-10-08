"""Market state of every test period (offline; usage: python3 market.py) -> MARKET.json.
For each benchmark and decision date: the holding-period return of the equal-weighted equity index of that benchmark
(PortBench: the equities in the date's universe, PortBench class map; InvestorBench: the four stocks, next-day return;
ClassAlloc: the equities class), and of the equal-weighted risky book (all non-cash assets)."""
import sys, json, pickle
from pathlib import Path
import numpy as np
HERE = Path(__file__).resolve().parent; ML = HERE.parent
sys.path.insert(0, str(ML / "portbench")); sys.path.insert(0, str(ML))
out = {}
# ClassAlloc
C = pickle.load(open(ML / "classalloc_and_robustness/classalloc/run/eval/cache.pkl", "rb"))
out["ClassAlloc"] = {"dates": C["dates"], "equity": [float(C["rel"][t][0, -1] - 1) for t in C["dates"]],
                     "risky_ew": [float(np.mean(C["rel"][t][:4, -1]) - 1) for t in C["dates"]]}
# InvestorBench
I = pickle.load(open(ML / "investorbench/turnover/cache.pkl", "rb"))
out["InvestorBench"] = {"dates": I["dates"], "equity": [float(np.mean(I["ret"][t])) for t in I["dates"]]}
out["InvestorBench"]["risky_ew"] = out["InvestorBench"]["equity"]
# PortBench
import monthly_lib as MLIB
rows, ids, dates, split = MLIB.load_monthly(); D = MLIB.Data(rows, ids, dates, split, cache=MLIB.load_cache())
RS = MLIB.RS; pb = {"dates": [], "equity": [], "risky_ew": [], "n_equity": [], "classes": {}}
for d in sorted(D.uni):
    uni = D.uni[d]; R = D.rel[d][:, -1] - 1; cl = [RS.cls(a) if callable(RS.cls) else RS.CMAP.get(a) for a in uni]
    eq = [r for r, c in zip(R, cl) if str(c).lower().startswith("equit")]; rk = [r for r, c in zip(R, cl) if str(c).lower() != "cash"]
    pb["dates"].append(d); pb["equity"].append(float(np.mean(eq)) if eq else None); pb["risky_ew"].append(float(np.mean(rk)))
    pb["n_equity"].append(len(eq))
    for c in cl: pb["classes"][str(c)] = pb["classes"].get(str(c), 0) + 1
pb["split"] = split; out["PortBench"] = pb
(HERE / "MARKET.json").write_text(json.dumps(out))
print({k: len(v["dates"]) for k, v in out.items()}, pb["classes"], pb["n_equity"][:5])
