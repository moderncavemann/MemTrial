"""Never-trust ablation on InvestorBench (offline, no API calls): ib_turnover.run_variant (frozen memtrial.py, unmodified)
with one added variant, alpha = 0; output written here, the published turnover/ files are not touched.
usage: python3 nt_ib.py   -> mt_NT_never_trust.json"""
import sys, json, time
from pathlib import Path
import numpy as np
HERE = Path(__file__).resolve().parent; LAB = HERE.parent
sys.path.insert(0, str(LAB / "investorbench")); sys.path.insert(0, str(LAB / "memtrial"))
import ib_turnover as IT
C = IT.load(); IT.VARIANTS["NT|never trust"] = dict(learner="auto", alpha=0.0); IT.OUT = HERE; t0 = time.time()
res, op = IT.run_variant("NT|never trust", C)
g = json.load(open(LAB / "investorbench/turnover/mt_MemTrial_gate_0_01.json"))
m = json.load(open(LAB / "investorbench/turnover/mt_MemTrial.json"))
for lab, R, O in (("NT|never trust", res, op), ("MemTrial | gate 0.01", np.array(g["res"]), np.array(g["open"])), ("MemTrial", np.array(m["res"]), np.array(m["open"]))):
    per = 1e4 * np.nanmean(np.asarray(R, float), (0, 1)); print(f"{lab:22s} {per.mean():.4f} +- {per.std(ddof=1):.4f} bp/day  trust {100*np.mean(O):.2f}%")
print("max |never trust - gate 0.01|:", float(np.nanmax(np.abs(res - np.array(g["res"])))), f"({time.time()-t0:.0f}s)")
