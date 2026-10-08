"""Half fraction vs full factorial (Appendix C.1; offline): on every date, seed and investor, the contribution c_j of Eq. (3) over the
8 designed subsets, the full-factorial Banzhaf value b_j over all 16 subsets, and the alias contrast a_j = c_j - b_j, which
is half the difference between the two half fractions and contains the three-way interaction aliased with experience j.
If that interaction is negligible, E[a_j^2] equals the noise term sigma^2/4, where sigma^2 is the variance of a draft's
utility across seeds (independent LLM draws of the same prompt). Utilities from the 1/N holdings path, as in Table 1.
usage: python3 alias_check.py cb|ib"""
import sys, json, pickle
from pathlib import Path
import numpy as np
HERE = Path(__file__).resolve().parent; LAB = HERE.parent
HALF = [0, 3, 5, 6, 9, 10, 12, 15]; ODD = [m for m in range(16) if m not in HALF]
bench = sys.argv[1]
if bench == "cb":
    sys.path.insert(0, str(LAB / "classalloc_and_robustness")); sys.path.insert(0, str(LAB / "memtrial")); import cb_eval as E
    C = pickle.load(open(LAB / "classalloc_and_robustness/classalloc/run/eval/cache.pkl", "rb")); R = C["rel"]; nxt = E.to_next(C["dates"])
    K = E.K; mv = lambda ew, t: E.drift(ew, nxt[t]); U = lambda x, t, g, h: float(E.util(x, R[t], g, h)[0][0]); scale = 100.0
else:
    sys.path.insert(0, str(LAB / "investorbench")); sys.path.insert(0, str(LAB / "memtrial")); import ib_turnover as E
    C = E.load(); R = C["ret"]; K = 5
    mv = lambda ew, t: E.drift(ew, R[t]); U = lambda x, t, g, h: float(E.util(x, R[t], g, h)[0]); scale = 1e4
W, seeds, dates = C["W"], C["seeds"], C["dates"]
out = {}
for p, (M, mfl, gam) in E.INV.items():
    ew = E.project(np.full(K, 1.0 / K), M, mfl); h = ew.copy(); A, Cc, Bf, S2 = [], [], [], []
    for t in dates:
        Us = np.array([[U(E.project(W[(t, s)][f"m{m}"], M, mfl), t, gam, h) for m in range(16)] for s in seeds]) * scale
        S2.append(Us.var(0, ddof=1).mean())
        for si in range(len(seeds)):
            u = Us[si]
            for j in range(4):
                on = lambda L: [m for m in L if (m >> j) & 1]; off = lambda L: [m for m in L if not (m >> j) & 1]
                c = u[on(HALF)].mean() - u[off(HALF)].mean(); co = u[on(ODD)].mean() - u[off(ODD)].mean(); b = (c + co) / 2
                A.append(c - b); Cc.append(c); Bf.append(b)
        h = mv(ew, t)
    A, Cc, Bf = map(np.array, (A, Cc, Bf)); s2 = float(np.mean(S2))
    out[p] = dict(mean_sq_alias=float((A ** 2).mean()), noise_quarter=s2 / 4, ratio=float((A ** 2).mean() / (s2 / 4)),
                  corr_half_full=float(np.corrcoef(Cc, Bf)[0, 1]), sd_contrib=float(Cc.std()), sd_alias=float(A.std()), n=len(A))
    print(bench, p, {k: (round(v, 4) if isinstance(v, float) else v) for k, v in out[p].items()}, flush=True)
json.dump(out, open(HERE / f"ALIAS_{bench}.json", "w"), indent=1)
