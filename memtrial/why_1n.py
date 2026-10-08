"""Why do LLM agents trail 1/N on the real benchmarks? (python3 why_1n.py pb | ib | pb0 | ib0) -> WHY_1N_<mode>.json
pb / ib : decomposition of utility (test period) into gross return, trading cost and risk penalty, plus the L1 distance
          from the equal-weight book (what the cost is charged on) and the risky share, for 1/N, zero-shot, similarity
          top-4 and the 8-draft average. Both benchmarks charge 15 bps on the distance from the equal-weight book.
pb0/ib0: all methods recomputed with zero trading cost (MemTrial included; its learning sees the zero-cost utilities).
Analysis only: published result files are not touched."""
import sys, os, json, collections, math
from pathlib import Path
import numpy as np
HERE = Path(__file__).resolve().parent; LAB = HERE.parent; sys.path.insert(0, str(HERE))
mode = sys.argv[1]; G3 = ("conservative", "balanced", "aggressive"); OUT = {}

if mode in ("pb", "pb0"):
    sys.path.insert(0, str(LAB / "portbench")); import pb_all as PA
    CM = PA.ML.CM
    if mode == "pb0":
        def U0(Q, W, G, prev, gamma):
            path = np.concatenate([np.ones((len(Q), 1)), Q @ G], 1); r = path[:, 1:] / path[:, :-1] - 1
            return path[:, -1] - 1 - 0.5 * gamma * r.var(1) * 20
        CM.U_batch = U0
    rows, ids, dates, split = PA.ML.load_monthly(); inp = json.load(open(PA.HERE / "data/inputs.json"))
    D = PA.ML.Data(rows, ids, dates, split, cache=PA.ML.load_cache()); Z = PA.content_features(D, inp)
    for c in ("full-price", "raw-price"):
        for p in G3:
            if mode == "pb0":
                res, _ = PA.run(D, c, p, Z); test = sorted(d for d in res["1/N"] if d >= split)
                OUT[f"{c}|{p}"] = {k: 100 * float(np.mean([np.mean(res[k][d]) for d in test])) for k in
                                   ["1/N", "Zero-shot (no memory)", "Similarity retrieval (top-4)", "Draft averaging (8 drafts)", "FinMem", "MemRL", "Reflexion",
                                    "Counterfactual selection (no gate)", "Hedge (memory families)", "MemTrial", "MemTrial | no gate"] if k in res}
                continue
            key = (c, p); gam = PA.ML.GAMMA[p]; acc = collections.defaultdict(lambda: collections.defaultdict(list))
            for d in sorted(D.proj[key]):
                if d < split: continue
                W = D.proj[key][d]; rel = D.rel[d]; prev = D.prev[d]
                arms = {"1/N": [D.ref[(d, p, "1/N")]]}
                for nm, view in (("zero-shot", "0"), ("similarity top-4", "15")):
                    arms[nm] = [W[r][view] for r in sorted(W) if view in W[r]]
                arms["draft average"] = [np.mean([W[r][m] for m in PA.MP.HALF if m in W[r]], 0) for r in sorted(W) if all(m in W[r] for m in PA.MP.HALF)]
                for nm, xs in arms.items():
                    for x in xs:
                        x = np.asarray(x) / np.sum(x); path = np.concatenate([[1.0], x @ rel]); rr = path[1:] / path[:-1] - 1
                        l1 = float(np.abs(x - prev)[np.abs(x - prev) >= 1e-6].sum()); nav = 1 - 0.0015 * l1
                        gross = path[-1] - 1; cost = gross - (nav * path[-1] - 1); risk = 0.5 * gam * np.var(nav * rr) * 20
                        u = float(D.U(x[None, :], d, p)[0])
                        for k, v in (("utility", u), ("gross return", gross), ("cost", cost), ("risk penalty", risk), ("L1 from equal weight", l1)):
                            acc[nm][k].append(v)
            OUT[f"{c}|{p}"] = {nm: {k: (100 * float(np.mean(v)) if k != "L1 from equal weight" else float(np.mean(v))) for k, v in kv.items()} for nm, kv in acc.items()}

if mode in ("ib", "ib0"):
    sys.path.insert(0, str(LAB / "investorbench")); import ib_all as IA, run_ib as RB
    if mode == "ib0":
        IA.FEE = 0.0; pub = IA.HERE / "ib_all.json"; keep = pub.read_bytes(); argv = sys.argv; sys.argv = [argv[0]]
        try:
            IA.main(); J = json.loads(pub.read_text())
        finally:
            pub.write_bytes(keep); sys.argv = argv
        for k in ["1/N", "Zero-shot (no memory)", "Similarity retrieval (top-4)", "Draft averaging", "FinMem", "MemRL", "Reflexion", "ExpeL",
                  "Counterfactual selection", "Hedge", "MemTrial", "MemTrial | no gate"]:
            x = np.array(J["res"][k], float); OUT[k] = {p: 1e4 * float(np.nanmean(x[:, i, :])) for i, p in enumerate(G3)}
    else:
        import glob
        B = RB.Bench(); recs = [json.load(open(f)) for f in glob.glob(str(IA.HERE / "run" / "executions" / "*.json"))]
        W = collections.defaultdict(dict)
        for r in recs:
            if r.get("phase") == "test" and r.get("status") == "VALID" and r["arm"] == "subset": W[(r["date"], r["seed"])][r["mask"]] = IA.vec(r["weights"])
        dates = [t for t in B.test if all(len(W.get((t, s), {})) == 16 for s in (0, 1, 2))]
        for p, (M, mfl, gam) in IA.INV.items():
            acc = collections.defaultdict(lambda: collections.defaultdict(list))
            for t in dates:
                r = np.array([B.next_returns(t)[a] / 100.0 for a in RB.ASSETS])
                for s in (0, 1, 2):
                    arms = {"1/N": IA.project(np.full(5, 0.2), M, mfl), "zero-shot": IA.project(W[(t, s)][0], M, mfl),
                            "similarity top-4": IA.project(W[(t, s)][15], M, mfl),
                            "draft average": np.mean([IA.project(W[(t, s)][m], M, mfl) for m in IA.HALF], 0)}
                    for nm, x in arms.items():
                        gross = float(x[:4] @ r); l1 = float(np.abs(x - 0.2).sum()); cost = 0.0015 * l1; net = gross - cost
                        for k, v in (("utility", net - 0.5 * gam * net ** 2), ("gross return", gross), ("cost", cost), ("risk penalty", 0.5 * gam * net ** 2),
                                     ("L1 from equal weight", l1), ("stock share", float(x[:4].sum()))):
                            acc[nm][k].append(v)
            OUT[p] = {nm: {k: (1e4 * float(np.mean(v)) if k in ("utility", "gross return", "cost", "risk penalty") else float(np.mean(v))) for k, v in kv.items()} for nm, kv in acc.items()}

(HERE / f"WHY_1N_{mode}.json").write_text(json.dumps(OUT, indent=1)); print(json.dumps(OUT, indent=1)[:6000])
