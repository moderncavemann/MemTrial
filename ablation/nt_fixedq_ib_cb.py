"""InvestorBench and ClassAlloc: the anchored action without online learning (q fixed at alpha0 = 0.9), offline.
IB: ib_turnover.run_variant (per-variant a0/lam0 supported); CA: cb_eval_extra.run_variant with the bank's and the
deployed blend's prior strength passed through (lam0 = 1e9 keeps q at alpha0). frozen memgate.py unmodified."""
import sys, json, pickle, inspect
from pathlib import Path
import numpy as np
HERE = Path(__file__).resolve().parent; LAB = HERE.parent
for p in (LAB / "memtrial", LAB / "investorbench", LAB / "classalloc_and_robustness"): sys.path.insert(0, str(p))
import ib_turnover as IT
C = IT.load(); IT.VARIANTS["NT|fixed q=0.9"] = dict(learner="auto", a0=0.9, lam0=1e9); IT.OUT = HERE
r, op = IT.run_variant("NT|fixed q=0.9", C)
m = json.load(open(LAB / "investorbench/turnover/mg_MemGate.json"))
for lab, R in (("IB MemGate", np.array(m["res"], float)), ("IB fixed q=0.9", r)):
    s = 1e4 * np.nanmean(R, (0, 1)); print(f"{lab:18s} {s.mean():.4f} ± {s.std(ddof=1):.4f}", np.round(s, 4))
q = np.array([x if x is not None else np.nan for x in np.array(m["q"], dtype=object).ravel()], float)
print("IB MemGate learned q: mean %.3f, first-day %.3f, last-day %.3f" % (np.nanmean(q), np.nanmean(np.array(m["q"], dtype=float)[0]), np.nanmean(np.array(m["q"], dtype=float)[-1])))
import cb_eval_extra as CX
src = inspect.getsource(CX.run_variant)
src = src.replace("bank = MemGateBank(Z, {name: cfg}, masks=CE.HALF)", "bank = MemGateBank(Z, {name: cfg}, masks=CE.HALF, a0=_A0, lam0=_L0)")
src = src.replace('ftrl_q(bank.sumG, bank.n, bank.sig, 0.5 if cl == "uniform" else 0.9, 4.0)', 'ftrl_q(bank.sumG, bank.n, bank.sig, 0.5 if cl == "uniform" else _A0, _L0)')
src = src.replace("    return dict(", "    _STORE[name] = (res.copy(), opened.copy(), list(qs))\n    return dict(", 1)
assert src.count("_A0") == 2 and "_STORE[name]" in src
CX._STORE = {}; exec(compile(src, "run_variant_fixedq", "exec"), CX.__dict__)
Cc = pickle.load(open(LAB / "classalloc_and_robustness/classalloc/run/eval/cache.pkl", "rb"))
CX._A0, CX._L0 = 0.9, 4.0; CX.run_variant(Cc, "MemGate")
pub = pickle.load(open(LAB / "classalloc_and_robustness/classalloc/run/eval/mg0.pkl", "rb"))[0]
print("CA identity vs published:", float(np.nanmax(np.abs(CX._STORE["MemGate"][0] - pub))))
CX.VARIANTS["NT|fixed q=0.9"] = dict(learner="auto"); CX._A0, CX._L0 = 0.9, 1e9; CX.run_variant(Cc, "NT|fixed q=0.9")
for k in ("MemGate", "NT|fixed q=0.9"):
    s = 100 * np.nanmean(CX._STORE[k][0], (0, 1)); qs = CX._STORE[k][2]
    print(f"CA {k:16s} {s.mean():.4f} ± {s.std(ddof=1):.4f}", np.round(s, 4), "mean q", round(float(np.mean(qs)), 3) if qs else None)
json.dump({"ib_fixed_q": r.tolist(), "ca": {k: CX._STORE[k][0].tolist() for k in CX._STORE}}, open(HERE / "nt_fixedq_ib_cb.json", "w"))
