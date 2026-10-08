"""Monthly net returns of every ClassAlloc method (offline; usage: python3 wealth_cb.py).

Runs the unchanged cb_eval.evaluate_base and cb_eval.evaluate_method('MemGate') on the cached logged drafts
(classalloc/run/eval/cache.pkl) once more. cb_eval.util is wrapped at run time (the file is not edited): the wrapper
returns the original outputs and stores utility -> net return J = nav * value - 1 (after the 15 bp fee on what is traded)
from the decision to the next decision date, where the account is rebalanced (cb_eval.to_next). Each method's utility is
then mapped to its J, so the wealth path is the value of the account. Check: the rerun utilities must equal the published
classalloc/run/eval/{base.pkl, mg0.pkl}. Writes WEALTH_CB.json.
Other LLMs: python3 wealth_cb.py run_<model>_t0.7  -> WEALTH_CB_<model>.json (same procedure on that run's eval files)."""
import sys, json, pickle, time
from pathlib import Path
import numpy as np
HERE = Path(__file__).resolve().parent; SUP = HERE.parent / "classalloc_and_robustness"
sys.path.insert(0, str(SUP))
import cb_eval as CE                      # unchanged

MAP = {}; CLASH = [0]; NXT = {}          # NXT: id of a date's 20-session relatives -> relatives to the next decision date
_orig = CE.util


def _rec(X, rel, gam, h):
    u, c, g = _orig(X, rel, gam, h)
    X2 = np.atleast_2d(X); nav = 1.0 - CE.FEE * np.abs(X2 - h).sum(1); J = nav * (X2 @ NXT[id(rel)])[:, -1] - 1.0
    for a, b in zip(np.atleast_1d(u), J):
        a, b = float(a), float(b)
        if a in MAP and abs(MAP[a] - b) > 1e-12: CLASH[0] += 1
        MAP[a] = b
    return u, c, g


CE.util = _rec


def main():
    t0 = time.time(); folder = sys.argv[1] if len(sys.argv) > 1 else "run"; ev = SUP / "classalloc" / folder / "eval"
    name = "WEALTH_CB.json" if folder == "run" else "WEALTH_CB_" + folder[4:].rsplit("_t", 1)[0] + ".json"
    C = pickle.load(open(ev / "cache.pkl", "rb")); nx = CE.to_next(C["dates"]); NXT.update({id(C["rel"][t]): nx[t] for t in C["dates"]})
    res, cost, held = CE.evaluate_base(C)
    mg = CE.evaluate_method(C, "MemGate")
    pub_base = pickle.load(open(ev / "base.pkl", "rb"))[0]; pub_mg = pickle.load(open(ev / "mg0.pkl", "rb"))[0]
    names = list(CE.MGV)
    assert names[0] == "MemGate", names
    res = dict(res); res["MemGate"] = mg[0]; pub = dict(pub_base); pub["MemGate"] = pub_mg
    worst = max(float(np.nanmax(np.abs(res[k] - pub[k]))) for k in res)
    net = {}; miss = 0
    for k, U in res.items():
        J = np.full(U.shape, np.nan)
        for idx, u in np.ndenumerate(U):
            if np.isnan(u): continue
            if float(u) in MAP: J[idx] = MAP[float(u)]
            else: miss += 1
        net[k] = J
    out = {"dates": C["dates"], "seeds": C["seeds"], "investors": list(CE.INV), "cutoff": "2024-07-01",
           "net": {k: np.where(np.isnan(v), None, v).tolist() for k, v in net.items()},
           "check": {"max_abs_diff_vs_published": worst, "unmapped": miss, "clashes": CLASH[0]}}
    (HERE / name).write_text(json.dumps(out))
    print(f"{time.time() - t0:.0f}s", out["check"], len(C["dates"]), "months")
    post = np.array([t >= "2024-07-01" for t in C["dates"]])
    for pi, p in enumerate(CE.INV):
        print(p)
        for k in ["1/N", "Zero-shot (no memory)", "FinMem", "MemRL", "Reflexion", "ExpeL", "MemGate"]:
            W = np.prod(1 + net[k][:, pi, :], 0); Wp = np.prod(1 + net[k][post, pi, :], 0)
            print(f"   {k:24s} all {W.mean():.3f} ± {W.std(ddof=1):.3f}   after cutoff {Wp.mean():.3f} ± {Wp.std(ddof=1):.3f}")


if __name__ == "__main__":
    main()
