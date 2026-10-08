"""ClassAlloc, MemTrial (Appendix D.11): on each decision where the trust test passed, d_t = u(deployed draft) - u(anchored blend it replaced),
both from the same holdings (cb_eval.evaluate_method, run with one added record; offline) -> TRUSTED_GAIN.json."""
import sys, pickle, json, math, inspect
import numpy as np
from pathlib import Path
HERE = Path(__file__).resolve().parent; sys.path.insert(0, str(HERE)); sys.path.insert(0, str(HERE.parent / "memtrial"))
import cb_eval as CE
run = sys.argv[1] if len(sys.argv) > 1 else "run"
C = pickle.load(open(HERE / "classalloc" / run / "eval" / "cache.pkl", "rb"))
src = inspect.getsource(CE.evaluate_method)
a = "                chk = float(util(X, R, gam, h)[0][0])\n"
b = ("                Xa = q * ew + (1 - q) * ens; DT.append((p, s, t, op, float(util(X, R, gam, h)[0][0]), float(util(Xa, R, gam, h)[0][0]), float(util(ew, R, gam, h)[0][0])))\n" + a)
assert src.count(a) == 1; src = src.replace(a, b)
ns = dict(CE.__dict__); ns["DT"] = []; exec(src, ns)
r = ns["evaluate_method"](C, "MemTrial")
pub = pickle.load(open(HERE / "classalloc" / run / "eval" / "mt0.pkl", "rb"))[0]
assert float(np.nanmax(np.abs(r[0] - pub))) == 0.0
DT = ns["DT"]; tr = [x for x in DT if x[3]]
d = 100 * np.array([x[4] - x[5] for x in tr]); dn = 100 * np.array([x[4] - x[6] for x in tr])
n = len(d); se = d.std(ddof=1) / math.sqrt(n)
out = {"run": run, "decisions": len(DT), "trusted": n, "mean_d_pp": float(d.mean()), "se_pp": float(se), "t": float(d.mean() / se),
       "share_positive": float(np.mean(d > 0)), "sum_d_pp_per_decision_all": float(d.sum() / len(DT)),
       "mean_vs_1N_pp": float(dn.mean()),
       "by_seed": {str(s): {"trusted": int(sum(1 for x in tr if x[1] == s)), "mean_d_pp": float(np.mean([100 * (x[4] - x[5]) for x in tr if x[1] == s])) if any(x[1] == s for x in tr) else None} for s in C["seeds"]},
       "by_year": {}}
for y in sorted({x[2][:4] for x in DT}):
    dy = [100 * (x[4] - x[5]) for x in tr if x[2][:4] == y]; ny = sum(1 for x in DT if x[2][:4] == y)
    out["by_year"][y] = {"trusted": len(dy), "decisions": ny, "mean_d_pp": float(np.mean(dy)) if dy else None}
(HERE / "classalloc" / run / "eval" / "TRUSTED_GAIN.json").write_text(json.dumps(out, indent=1))
print(json.dumps({k: v for k, v in out.items() if k != "by_year"}, indent=1)); print({y: (v["trusted"], v["decisions"], None if v["mean_d_pp"] is None else round(v["mean_d_pp"], 3)) for y, v in out["by_year"].items()})
