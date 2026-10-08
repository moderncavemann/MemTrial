"""Case study on ClassAlloc (offline; python3 case_cb.py INVESTOR DATE [DATE ...]) -> CASE_CB_<investor>.json.
Replays MemTrial (frozen memtrial.py, the loop of cb_eval.evaluate_method) for every seed and records, on the given dates,
what the method saw and did: the retrieved lessons, the eight designed drafts and their utilities, the Banzhaf contributions,
the learned values, the forward-score trust test, the action and its outcome. Read-only on published files."""
import sys, json, pickle
from pathlib import Path
import numpy as np
HERE = Path(__file__).resolve().parent; LAB = HERE.parent; SUP = LAB / "classalloc_and_robustness"
sys.path.insert(0, str(SUP)); sys.path.insert(0, str(LAB / "memtrial"))
import memtrial as MT
from memtrial import MemTrialBank, GRID, ftrl_q, banzhaf
import cb_eval as CE, run_cb as RC
inv, targets = sys.argv[1], sys.argv[2:]
C = pickle.load(open(SUP / "classalloc/run/eval/cache.pkl", "rb"))
W, top4, seeds, dates, rel, Z = C["W"], C["top4"], C["seeds"], C["dates"], C["rel"], C["Z"]
exps = {e["id"]: e for e in json.load(open(SUP / "classalloc/run/experiences.json"))}
B = RC.Bench(); M_, mfl, gam = CE.INV[inv]; ew = CE.project(np.full(CE.K, 1.0 / CE.K), M_, mfl)
out = {"investor": inv, "classes": CE.CL, "one_over_n": ew.tolist(), "cases": {}}
nxt = CE.to_next(dates)
for s in seeds:
    bank = MemTrialBank(Z, {"MemTrial": dict(learner="auto")}, masks=CE.HALF); h = ew.copy(); fb = CE.Feedback()
    for t in dates:
        for a_ in fb.ready(t): bank.matured(*a_)
        w = {k: CE.project(v, M_, mfl) for k, v in W[(t, s)].items()}; R = rel[t]; ids = top4[t]
        ens = np.mean([w[f"m{m}"] for m in CE.HALF], 0)
        U = {m: float(CE.util(w[f"m{m}"], R, gam, h)[0][0]) for m in CE.HALF}
        u_ref = float(CE.util(ew, R, gam, h)[0][0]); u_ens = float(CE.util(ens, R, gam, h)[0][0])
        grid = CE.util(GRID[:, None] * ew[None, :] + (1 - GRID)[:, None] * ens[None, :], R, gam, h)[0]
        q = ftrl_q(bank.sumG, bank.n, bank.sig, 0.9, 4.0)
        out_u = bank.decide(t, ids, U, u_ref, u_ens, grid)["MemTrial"]; op = bool(bank.log["MemTrial"][-1][1]); p = float(bank.log["MemTrial"][-1][2])
        sc = {m: (np.mean(bank.G[m].scores) if len(bank.G[m].scores) >= MT.MIN_SCORES else -np.inf) for m in ("identity", "content")}
        lm = "content" if sc["content"] > sc["identity"] else "identity"
        v = [float(bank.L[lm].predict(x)) for x in ids]
        best = max(CE.HALF, key=lambda k: (sum(v[j] for j in range(len(ids)) if (k >> j) & 1), -bin(k).count("1")))
        X = w[f"m{best}"] if op else q * ew + (1 - q) * ens
        if t in targets:
            c = [float(banzhaf(U, j, CE.HALF)) for j in range(4)]
            out["cases"][f"{t}|seed{s}"] = {
                "date": t, "seed": s, "context": B.context(t),
                "lessons": [{"id": x, "date": exps[x].get("date"), "text": exps[x].get("text")} for x in ids],
                "drafts": {str(m): {"lessons": [j + 1 for j in range(4) if (m >> j) & 1], "weights": [round(float(a), 3) for a in w[f"m{m}"]],
                                    "utility_pp": round(100 * U[m], 3)} for m in CE.HALF},
                "contributions_pp": [round(100 * x, 3) for x in c], "values_pp": [round(100 * x, 3) for x in v], "learner": lm,
                "forward_scores": {m: (len(bank.G[m].scores), float(np.mean(bank.G[m].scores)) if bank.G[m].scores else None) for m in ("identity", "content")},
                "p_value": p, "trusted": op, "best_subset": [j + 1 for j in range(4) if (best >> j) & 1], "q_on_1N": float(q),
                "deployed": [round(float(a), 3) for a in X], "utility_pp": round(100 * out_u, 3), "one_over_n_utility_pp": round(100 * u_ref, 3),
                "draft_average_utility_pp": round(100 * u_ens, 3), "realized_class_returns_pct": [round(100 * (float(r) - 1), 2) for r in R[:, -1]]}
        fb.add(t, (t, ids, U, u_ref, u_ens, grid)); h = CE.drift(X, nxt[t])
(HERE / f"CASE_CB_{inv}.json").write_text(json.dumps(out, indent=1))
for k, v in out["cases"].items():
    print(k, "trusted" if v["trusted"] else "anchored", "p=%.3f" % v["p_value"], "q=%.2f" % v["q_on_1N"], "u=%.2f vs 1/N %.2f" % (v["utility_pp"], v["one_over_n_utility_pp"]),
          "values", v["values_pp"], "contrib", v["contributions_pp"], "best", v["best_subset"])
