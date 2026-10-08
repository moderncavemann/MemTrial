"""All methods on the PortBench monthly extension with per-seed values (usage: python3 pb_all.py <cfg> <investor>)
-> pb_all_<cfg>_<investor>.json  {method: {date: [seed0, seed1, seed2]}}.
Seeds = the three independent draws per arm (deployment worlds). Time-forward: a decision at date d uses only dates
before d. MemTrial is the frozen version in ../memtrial (MEMTRIAL_FROZEN.json): designed subsets = the 8 half-fraction
subsets; content features = 8-dim PCA of the Qwen3 retrieval embeddings of the experiences."""
import sys, json, math, random, collections
from pathlib import Path
import numpy as np
HERE = Path(__file__).resolve().parent; sys.path.insert(0, str(HERE)); sys.path.insert(0, str(HERE.parent / "memtrial"))
import monthly_lib as ML, monthly_policies as MP
from memtrial import MemTrialBank
HALF = [int(m) for m in MP.HALF]; MASKS = MP.MASKS
VARIANTS = {"MemTrial": dict(learner="auto"), "MemTrial | content learner only": dict(learner="content"),
            "MemTrial | identity learner": dict(learner="identity"), "MemTrial | F-test gate": dict(learner="auto", gate="ftest"),
            "MemTrial | no gate": dict(learner="auto", gate="none"), "MemTrial | closed->ensemble": dict(learner="auto", closed="ens"),
            "MemTrial | closed->reference": dict(learner="auto", closed="ref"), "MemTrial | uniform prior": dict(learner="auto", closed="uniform"),
            "MemTrial | gate 0.2": dict(learner="auto", alpha=0.2), "MemTrial | gate 0.01": dict(learner="auto", alpha=0.01)}


def content_features(D, inp):
    """8-dim PCA (standardised) of the retrieval embeddings (Qwen3-Embedding-0.6B) of the experiences retrieved on inp's dates."""
    emb = json.load(open(HERE / "data" / "experience_embeddings.json"))
    ids = sorted({h for d in inp["dates"] for h in inp["ids"][d]}); X = np.array([emb[h] for h in ids], float)
    X = X - X.mean(0); U, S, Vt = np.linalg.svd(X, full_matrices=False); Zc = X @ Vt[:8].T; Zc = Zc / Zc.std(0)
    return {h: list(z) for h, z in zip(ids, Zc)}


def run(D, cfg, p, Z):
    key = (cfg, p); res = collections.defaultdict(lambda: collections.defaultdict(list)); gate_log = []
    valid = [d for d in D.dates if d in D.proj[key] and all(any(m in D.proj[key][d][r] for r in D.proj[key][d]) for m in MASKS)]
    fams = [f for f in MP.FAMILIES if all(any(f in D.proj[key][d][r] for r in D.proj[key][d]) for d in valid)]
    for r in (0, 1, 2):
        hist = []; up_obs = []; votes = collections.defaultdict(float); rnd = random.Random(f"{cfg}|{p}|{r}")
        bank = MemTrialBank(Z, VARIANTS, masks=HALF)
        for d in valid:
            W = D.proj[key][d]
            def X(arm):
                if arm in W.get(r, {}): return W[r][arm]
                for rr in sorted(W):
                    if arm in W[rr]: return W[rr][arm]
            Xs = {m: X(m) for m in MASKS}; Xs.update({f: X(f) for f in fams if f not in Xs})
            ref, mv = D.ref[(d, p, "1/N")], D.ref[(d, p, "MinVar")]
            ens8 = np.mean([Xs[m] for m in MP.HALF], 0); ens16 = np.mean([Xs[m] for m in MASKS], 0)
            names = ["1/N", "MinVar"] + MASKS + [f for f in fams if f not in MASKS] + ["ENS8", "ENS16"]
            u = dict(zip(names, D.U(np.array([ref, mv] + [Xs[m] for m in MASKS] + [Xs[f] for f in fams if f not in MASKS] + [ens8, ens16]), d, p)))
            ids = D.ids[d]
            out = {"1/N": u["1/N"], "Minimum variance": u["MinVar"], "Zero-shot (no memory)": u["0"], "Similarity retrieval (top-4)": u["15"],
                   "Similarity retrieval (top-2)": u["3"], "Draft averaging (8 drafts)": u["ENS8"], "Draft averaging (16 drafts)": u["ENS16"],
                   "FinMem": u["FinMem"], "MemRL": u["MemRL"], "Oracle (best arm in hindsight)": max(u[m] for m in MASKS + ["FinMem", "MemRL", "M2_original"])}
            if X("Reflexion") is not None: out["Reflexion"] = float(D.U(X("Reflexion"), d, p)[0])
            vs = [W[rr]["0"] for rr in sorted(W) if "0" in W[rr]]; out["Self-consistency (3 drafts)"] = float(D.U(np.mean(vs, 0), d, p)[0])
            # ---- learned selectors over the 4 retrieved experiences (time-forward; this world's draws) ----
            if hist:
                cf = collections.defaultdict(list)
                for h in hist:
                    for j, hid in enumerate(h["ids"]): cf[hid].append(h["cf"][j])
                sc = [np.mean(cf[h]) if h in cf else 0.0 for h in ids]; o = sorted(range(4), key=lambda j: (-sc[j], j))
                out["Counterfactual selection (no gate)"] = u[str((1 << o[0]) | (1 << o[1]))]
            else:
                out["Counterfactual selection (no gate)"] = u["3"]
            if len(up_obs) >= 3:                                    # uplift credit: regression-adjusted (UpliftMem/UCOB-style)
                allid = sorted({h for o_ in up_obs for h in o_[0]} | set(ids)); ix = {h: i for i, h in enumerate(allid)}
                Xr = np.array([[1.0, o_[2]] + [1.0 if h in o_[0] else 0.0 for h in allid] for o_ in up_obs]); yr = np.array([o_[1] for o_ in up_obs])
                pen = np.eye(Xr.shape[1]); pen[0, 0] = pen[1, 1] = 0.0
                coef = np.linalg.solve(Xr.T @ Xr + 1e-4 * pen * len(yr), Xr.T @ yr); upl = [coef[2 + ix[h]] for h in ids]
            else:
                upl = [0.0] * 4
            o = rnd.sample(range(4), 2) if rnd.random() < 0.1 else sorted(range(4), key=lambda j: (-upl[j], j))[:2]
            mk = str((1 << o[0]) | (1 << o[1])); out["Uplift credit (UpliftMem-style)"] = u[mk]
            up_obs.append(({ids[o[0]], ids[o[1]]}, u[mk], u["1/N"]))
            o = sorted(range(4), key=lambda j: (-votes[ids[j]], j))[:2]   # ExpeL-style insight votes (adapted)
            mk = str((1 << o[0]) | (1 << o[1])); out["ExpeL (adapted)"] = u[mk]
            for j in o: votes[ids[j]] += 1.0 if u[mk] > u["1/N"] else -1.0
            exps = ["1/N"] + fams; Wf = np.array([ref] + [Xs[f] for f in fams])
            qh = MP.hedge([h["fam_u"] for h in hist], np.full(len(exps), 1 / len(exps)))
            out["Hedge (memory families)"] = float(D.U(qh @ Wf, d, p)[0])
            # ---- MemTrial (frozen) and ablations ----
            Uh = {m: u[str(m)] for m in HALF}; grid = MP.ugrid(D.pack(np.array([ref, ens8]), d, p))
            out.update(bank.decide(d, ids, Uh, u["1/N"], u["ENS8"], grid))
            gate_log.append({"world": r, "date": d, "open": bool(bank.log["MemTrial"][-1][1]), "p": float(bank.log["MemTrial"][-1][2])})
            for k, v in out.items(): res[k][d].append(float(v))
            bank.matured(d, ids, Uh, u["1/N"], u["ENS8"], grid)
            T = {m: u[m] for m in MASKS}
            hist.append({"ids": ids, "cf": [MP.banzhaf(T, j) for j in range(4)], "fam_u": [u["1/N"]] + [u[f] for f in fams]})
    return {k: dict(v) for k, v in res.items()}, gate_log


if __name__ == "__main__":
    cfg, p = sys.argv[1], sys.argv[2]
    rows, ids, dates, split = ML.load_monthly(); inp = json.load(open(HERE / "data/inputs.json"))
    D = ML.Data(rows, ids, dates, split, cache=ML.load_cache())
    Z = content_features(D, inp)
    res, gl = run(D, cfg, p, Z)
    json.dump({"res": res, "gate": gl, "split": split}, open(HERE / f"pb_all_{cfg}_{p}.json", "w"))
    test = [d for d in res["1/N"] if d >= split]
    for k in ["1/N", "Zero-shot (no memory)", "FinMem", "Reflexion", "Uplift credit (UpliftMem-style)", "ExpeL (adapted)", "MemTrial", "MemTrial | F-test gate"]:
        per = [np.mean([res[k][d][w] for d in test]) for w in range(3)]
        print(f"{k:34s} test {100*np.mean(per):.3f} +- {100*np.std(per, ddof=1):.3f}")
    print("gate open (test):", np.mean([g["open"] for g in gl if g["date"] >= split]))
