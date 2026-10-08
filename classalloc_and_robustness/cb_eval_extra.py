#!/usr/bin/env python3
"""ClassAlloc: ablation variants of the method (Figure 4 rows) and the knowledge-cutoff contrast; offline, no API calls.
Reads classalloc/<run>/eval/cache.pkl (cb_eval.py --stage cache) and writes classalloc/<run>/eval/EXTRA.json.
Usage: python3 cb_eval_extra.py [--run FOLDER] [--part ablation|cutoff|all]"""
import json, math, pickle, sys
from pathlib import Path
import numpy as np
HERE = Path(__file__).resolve().parent; sys.path.insert(0, str(HERE)); sys.path.insert(0, str(HERE.parent / "memtrial"))
import memgate as MG
from memgate import MemGateBank, GRID, ftrl_q
import cb_eval as CE

VARIANTS = {"MemGate": dict(learner="auto"),
            "identity learner only": dict(learner="identity"), "content learner only": dict(learner="content"),
            "F-test gate": dict(learner="auto", gate="ftest"), "no gate": dict(learner="auto", gate="none"),
            "closed -> draft average": dict(learner="auto", closed="ens"), "closed -> reference": dict(learner="auto", closed="ref"),
            "uniform prior": dict(learner="auto", closed="uniform")}


def run_variant(C, name):
    cfg = VARIANTS[name]; W, top4, seeds, dates, rel, Z = C["W"], C["top4"], C["seeds"], C["dates"], C["rel"], C["Z"]
    nD, nS = len(dates), len(seeds); res = np.full((nD, 3, nS), np.nan); opened = np.zeros((nD, 3, nS)); qs = []
    cl = cfg.get("closed", "ftrl"); nxt = CE.to_next(dates)
    for pi, (p, (M, mfl, gam)) in enumerate(CE.INV.items()):
        ew = CE.project(np.full(CE.K, 1.0 / CE.K), M, mfl)
        for si, s in enumerate(seeds):
            bank = MemGateBank(Z, {name: cfg}, masks=CE.HALF); h = ew.copy(); fb = CE.Feedback()
            for di, t in enumerate(dates):
                for a_ in fb.ready(t): bank.matured(*a_)
                w = {k: CE.project(v, M, mfl) for k, v in W[(t, s)].items()}; R = rel[t]; ids = top4[t]
                ens = np.mean([w[f"m{m}"] for m in CE.HALF], 0)
                U = {m: float(CE.util(w[f"m{m}"], R, gam, h)[0][0]) for m in CE.HALF}
                u_ref = float(CE.util(ew, R, gam, h)[0][0]); u_ens = float(CE.util(ens, R, gam, h)[0][0])
                grid = CE.util(GRID[:, None] * ew[None, :] + (1 - GRID)[:, None] * ens[None, :], R, gam, h)[0]
                q = ftrl_q(bank.sumG, bank.n, bank.sig, 0.5 if cl == "uniform" else 0.9, 4.0)
                out = bank.decide(t, ids, U, u_ref, u_ens, grid)[name]; op = bool(bank.log[name][-1][1])
                if op:
                    lm = cfg.get("learner", "content")
                    if lm == "auto":
                        sc = {m: np.mean(bank.G[m].scores) if len(bank.G[m].scores) >= MG.MIN_SCORES else -np.inf for m in ("identity", "content")}
                        lm = "content" if sc["content"] > sc["identity"] else "identity"
                    v = [bank.L[lm].predict(hh) for hh in ids]
                    best = max(CE.HALF, key=lambda k: (sum(v[j] for j in range(len(ids)) if (k >> j) & 1), -bin(k).count("1")))
                    X = w[f"m{best}"]
                elif cl == "ens": X = ens
                elif cl == "ref": X = ew
                else: X = q * ew + (1 - q) * ens; qs.append(q)
                chk = float(CE.util(X, R, gam, h)[0][0])
                if abs(chk - out) > 1e-6: raise RuntimeError(f"mismatch {name} {p} {s} {t}")
                fb.add(t, (t, ids, U, u_ref, u_ens, grid))
                res[di, pi, si] = out; opened[di, pi, si] = op; h = CE.drift(X, nxt[t])
    per_seed = 100 * res.mean((0, 1))
    return dict(mean=float(per_seed.mean()), sd=float(per_seed.std(ddof=1)), trust_pct=100 * float(opened.mean()),
                mean_q_closed=float(np.mean(qs)) if qs else None, by_date=(100 * res.mean((1, 2))).tolist())


def welch(a, b):
    a, b = np.asarray(a), np.asarray(b); va, vb = a.var(ddof=1) / len(a), b.var(ddof=1) / len(b)
    t = (a.mean() - b.mean()) / math.sqrt(va + vb); df = (va + vb) ** 2 / (va ** 2 / (len(a) - 1) + vb ** 2 / (len(b) - 1))
    return float(t), float(df), float(2 * MG.t_sf(abs(t), df))


def main():
    name = sys.argv[sys.argv.index("--run") + 1] if "--run" in sys.argv else "run"
    part = sys.argv[sys.argv.index("--part") + 1] if "--part" in sys.argv else "all"
    ev = HERE / "classalloc" / name / "eval"; C = pickle.load(open(ev / "cache.pkl", "rb"))
    path = ev / "EXTRA.json"; out = json.loads(path.read_text()) if path.exists() else {}
    if part in ("all", "ablation"):
        out["ablation"] = {}
        for n in VARIANTS:
            r = run_variant(C, n); out["ablation"][n] = {k: v for k, v in r.items() if k != "by_date"}
            print(f"{n:28s} {r['mean']:.3f} ± {r['sd']:.3f} pp; trust {r['trust_pct']:.1f}%; mean q (closed) {r['mean_q_closed']}", flush=True)
    if part in ("all", "cutoff"):
        R = json.load(open(ev / "RESULTS.json")); res = {k: np.array(v) for k, v in R["res_pp"].items()}; dates = R["dates"]
        post = np.array([t >= "2024-07-01" for t in dates]); cut = {}
        for k in ["Zero-shot (no memory)", "Self-consistency", "Draft averaging", "Similarity retrieval (top-4)", "MemRL", "MemGate"]:
            d = res[k].mean((1, 2)) - res["1/N"].mean((1, 2))          # per month, mean over investors and seeds
            t, df, p = welch(d[~post], d[post])
            cut[k] = dict(before=float(d[~post].mean()), after=float(d[post].mean()), welch_t=t, df=df, p_two_sided=p)
            print(f"{k:30s} minus 1/N: before {d[~post].mean():+.3f} pp ({(~post).sum()} months), after {d[post].mean():+.3f} pp ({post.sum()}); Welch p = {p:.3f}")
        out["cutoff_contrast_vs_1N"] = cut
    path.write_text(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
