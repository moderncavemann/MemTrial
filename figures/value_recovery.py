"""PlantedMem value recovery (analysis only; no LLM call, no change to any existing file). Replays the PlantedMem episode of
plantedmem/suite.py exactly (its source is reused verbatim and only logging lines are added at run time), and records
on every test date, from matured dates only, the value each method has learned for every experience retrieved so far:
  outcome credit  - MemRL-style running value Q (the 'outcome credit top-2' policy of the suite),
  counterfactual  - mean past Banzhaf contribution (what Counterfactual selection ranks by),
  MemTrial        - posterior mean of the selected value learner (frozen memtrial.py, learner chosen by forward scores).
Score per date: Spearman correlation between these values and the planted quality (+1 useful, 0 neutral, -1 misleading).
usage: python3 value_recovery.py run <budget_s>  -> VR_LOG.jsonl (resumable); python3 value_recovery.py summary -> VR_SUMMARY.json"""
import sys, json, time, inspect
from pathlib import Path
import numpy as np
HERE = Path(__file__).resolve().parent; LAB = HERE.parent
for p in (LAB / "plantedmem", LAB / "plantedmem", LAB / "memtrial"): sys.path.insert(0, str(p))
import suite as S5
import memtrial as MTm
REGS = ["beta 0.5", "beta 0.25", "no influence", "many experiences, informative content"]
SEEDS = range(15, 115); GAMMA = 5.0                       # test seeds, balanced investor
src = inspect.getsource(S5.episode).replace("def episode(", "def episode_log(", 1)
a1 = "    for t in range(len(D)):\n"
assert src.count(a1) == 1; src = src.replace(a1, "    LOG = []\n" + a1 + "        q_pre = Q.copy()\n", 1)
a2 = "        if t >= warm:\n            for k, v in out.items(): res[k].append(v)\n"
assert src.count(a2) == 1
inj = ("        if t >= warm:\n"
       "            sc_ = {mm: np.mean(MT.G[mm].scores) if len(MT.G[mm].scores) >= MTm.MIN_SCORES else -np.inf for mm in ('identity', 'content')}\n"
       "            lm_ = 'content' if sc_['content'] > sc_['identity'] else 'identity'\n"
       "            LOG.append(dict(t=t, seen=[m for m in range(M) if len(obs[m]) > 0], mt=[MT.L[lm_].predict(m) for m in range(M)],\n"
       "                            cf=[float(np.mean(obs[m])) if obs[m] else 0.0 for m in range(M)], oc=[float(x) for x in q_pre],\n"
       "                            open=float(MT.log['MemTrial'][-1][1]), learner=lm_))\n")
src = src.replace(a2, inj + a2, 1)
a3 = "    means = {k: float(np.mean(v))"
assert src.count(a3) == 1; src = src.replace(a3, "    return LOG, [int(x) for x in types]\n" + a3, 1)
NS = S5.__dict__; NS["MTm"] = MTm; exec(src, NS); episode_log = NS["episode_log"]


def ranks(x):
    x = np.asarray(x, float); o = np.argsort(x, kind="mergesort"); r = np.empty(len(x)); r[o] = np.arange(len(x), dtype=float)
    for v in np.unique(x):
        idx = np.flatnonzero(x == v)
        if len(idx) > 1: r[idx] = r[idx].mean()
    return r


def spearman(a, b):
    ra, rb = ranks(a), ranks(b)
    if ra.std() == 0 or rb.std() == 0: return 0.0
    return float(np.corrcoef(ra, rb)[0, 1])


if __name__ == "__main__":
    out = HERE / "VR_LOG.jsonl"
    if sys.argv[1] == "run":
        budget = float(sys.argv[2]); t0 = time.time()
        done = set()
        if out.exists():
            for line in out.read_text().splitlines():
                J = json.loads(line); done.add((J["regime"], J["seed"]))
        n_new = 0
        for n in REGS:
            for s in SEEDS:
                if (n, s) in done: continue
                if time.time() - t0 > budget: print("budget reached; new", n_new, "done", len(done)); sys.exit(0)
                LOG, types = episode_log(seed=s, gamma=GAMMA, **S5.REGIMES[n])
                rho = {k: [] for k in ("oc", "cf", "mt")}; opened = []
                for e in LOG:
                    seen = e["seen"]; q = [types[m] for m in seen]
                    for k in rho: rho[k].append(spearman([e[k][m] for m in seen], q) if len(seen) >= 3 else 0.0)
                    opened.append(e["open"])
                with open(out, "a") as f: f.write(json.dumps({"regime": n, "seed": s, "rho": rho, "open": opened}) + "\n")
                done.add((n, s)); n_new += 1
        print("all done", len(done))
    elif sys.argv[1] == "summary":
        R = [json.loads(l) for l in out.read_text().splitlines()]
        S = {}
        for n in REGS:
            rows = [r for r in R if r["regime"] == n]
            if not rows: continue
            S[n] = {"n_seeds": len(rows)}
            for k in ("oc", "cf", "mt"):
                A = np.array([r["rho"][k] for r in rows]); S[n][k] = {"mean": A.mean(0).tolist(), "sd": A.std(0, ddof=1).tolist(),
                                                                    "last12_mean": float(A[:, -12:].mean()), "first12_mean": float(A[:, :12].mean())}
            S[n]["open"] = np.array([r["open"] for r in rows]).mean(0).tolist()
        (HERE / "VR_SUMMARY.json").write_text(json.dumps(S)); print({n: {k: (round(v["first12_mean"], 3), round(v["last12_mean"], 3)) for k, v in S[n].items() if isinstance(v, dict)} for n in S})
