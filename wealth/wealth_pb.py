"""Monthly net returns of every PortBench method (offline; usage: python3 wealth_pb.py <cfg> <investor>).

Runs the unchanged pb_all.run (frozen MemTrial code, same logged drafts) once more and records, for every portfolio that
the unchanged code evaluates, its net holding-period return J = nav * value_20 - 1 (the J inside mixture.U_batch, i.e.
after the 15 bp fee on turnover). mixture.U_batch is wrapped at run time (the file is not edited): the wrapper returns
the original utilities unchanged and stores utility -> J. Each method's stored utility is then mapped to the J of the
portfolio that produced it. Check: the rerun utilities must equal the published pb_all_<cfg>_<investor>.json.
Writes WEALTH_PB_<cfg>_<investor>.json."""
import sys, json, os, time
from pathlib import Path
import numpy as np
HERE = Path(__file__).resolve().parent; MON = HERE.parent / "portbench"
sys.path.insert(0, str(MON)); sys.path.insert(0, str(HERE.parent / "memtrial"))
import pb_all as PA                      # unchanged
import monthly_lib as MLIB
import mixture as CM

MAP = {}; CLASH = [0]
_orig = CM.U_batch


def _rec(Q, W, G, prev, gamma):
    u = _orig(Q, W, G, prev, gamma)
    Wm = Q @ W; ad = np.abs(Wm - prev); to = np.where(ad >= 1e-6, ad, 0).sum(1)
    J = (1 - 0.0015 * to) * (Q @ G)[:, -1] - 1
    for a, b in zip(np.atleast_1d(u), J):
        a, b = float(a), float(b)
        if a in MAP and abs(MAP[a] - b) > 1e-12: CLASH[0] += 1
        MAP[a] = b
    return u


CM.U_batch = _rec


def main():
    cfg, p = sys.argv[1], sys.argv[2]; t0 = time.time()
    rows, ids, dates, split = MLIB.load_monthly(); inp = json.load(open(MON / "data/inputs.json"))
    D = MLIB.Data(rows, ids, dates, split, cache=MLIB.load_cache())
    Z = PA.content_features(D, inp)
    res, gl = PA.run(D, cfg, p, Z)
    pub = json.load(open(MON / f"pb_all_{cfg}_{p}.json"))["res"]
    worst = 0.0; miss = 0; net = {}
    for k, byd in res.items():
        net[k] = {}
        for d, us in byd.items():
            if k in pub and d in pub[k]: worst = max(worst, max(abs(a - b) for a, b in zip(us, pub[k][d])))
            js = []
            for u in us:
                if u in MAP: js.append(MAP[u])
                else: js.append(None); miss += 1
            net[k][d] = js
    out = {"cfg": cfg, "investor": p, "split": split, "dates": sorted(res["1/N"]), "net": net, "util": res,
           "check": {"max_abs_diff_vs_published": worst, "unmapped": miss, "clashes": CLASH[0],
                     "methods_missing_in_published": sorted(set(res) - set(pub))}}
    (HERE / f"WEALTH_PB_{cfg}_{p}.json").write_text(json.dumps(out))
    test = [d for d in out["dates"] if d >= split]
    print(cfg, p, f"{time.time() - t0:.0f}s", out["check"], len(test), "test months")
    for k in ["1/N", "Zero-shot (no memory)", "FinMem", "MemRL", "Reflexion", "ExpeL (adapted)", "MemTrial"]:
        W = np.prod([1 + np.array(net[k][d], float) for d in test], 0)
        print(f"   {k:26s} terminal wealth {W.mean():.4f} ± {W.std(ddof=1):.4f}")


if __name__ == "__main__":
    main()
