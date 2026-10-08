"""Never-trust ablation on ClassAlloc (offline, no API calls): cb_eval.evaluate_method (frozen memtrial.py, unmodified)
with one added variant, alpha = 0. Also re-runs MemTrial and checks that it reproduces the published mt0.pkl exactly.
usage: python3 nt_cb.py [run]   -> nt_cb_<run>.json"""
import sys, json, pickle, time
from pathlib import Path
import numpy as np
HERE = Path(__file__).resolve().parent; LAB = HERE.parent
sys.path.insert(0, str(LAB / "classalloc_and_robustness")); sys.path.insert(0, str(LAB / "memtrial"))
import cb_eval as CE
CE.MTV["NT|never trust"] = dict(learner="auto", alpha=0.0)
name = sys.argv[1] if len(sys.argv) > 1 else "run"; ev = CE.HERE / "classalloc" / name / "eval"
C = pickle.load(open(ev / "cache.pkl", "rb")); out = {"dates": C["dates"], "seeds": C["seeds"]}; t0 = time.time()
for n in ("MemTrial", "NT|never trust"):
    r, c, op, fs = CE.evaluate_method(C, n); out[n] = {"res": r.tolist(), "open": op.tolist()}
pub = pickle.load(open(ev / "mt0.pkl", "rb"))[0]
out["identity_max_abs_diff_vs_published_MemTrial"] = float(np.nanmax(np.abs(np.array(out["MemTrial"]["res"]) - pub)))
base = pickle.load(open(ev / "base.pkl", "rb"))[0]
for k in ("1/N", "Draft averaging"):
    if k in base: out[k] = {"res": np.asarray(base[k]).tolist()}
json.dump(out, open(HERE / f"nt_cb_{name}.json", "w"))
for k in ("1/N", "Draft averaging", "MemTrial", "NT|never trust"):
    if k in out:
        per = 100 * np.nanmean(np.array(out[k]["res"], float), (0, 1)); op = np.array(out[k].get("open", [[[0]]]), float).mean()
        print(f"{k:22s} {per.mean():.4f} +- {per.std(ddof=1):.4f} pp/month  trust {100*op:.1f}%")
print("identity vs published MemTrial:", out["identity_max_abs_diff_vs_published_MemTrial"], f"({time.time()-t0:.0f}s)")
