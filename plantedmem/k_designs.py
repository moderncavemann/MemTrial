"""How many retrieved experiences (k)? PlantedMem sensitivity of MemTrial (python3 k_designs.py run <investor> <s0> <s1>
| prun <investor> <s0> <s1> | table <s0> <s1>) -> results/k_<investor>_<s0>_<s1>.json, results/K_TABLE.md.
Same worlds as suite (planted experiences, regime-correlated retrieval, drafts resampled from logged LLM allocations).
The retriever returns the 6 best matches; MemTrial uses the first k of them with a designed set of subsets:
k=2 full factorial (4 drafts), k=3 half fraction (4) and full factorial (8), k=4 half fraction (8, the default),
k=5 half fraction (16), k=6 half fraction (32). All designs share the retrieved experiences, the experiences' predictions,
the draft draws and the reference draws (common random numbers), so differences come from k and the design only.
The streams differ from suite, so the k=4 row is close to, not identical with, Table 3. memtrial.py is unmodified."""
import sys, math, random, collections, json, time, glob
from pathlib import Path
import numpy as np
HERE = Path(__file__).resolve().parent; sys.path.insert(0, str(HERE))
import suite as S
from memtrial import MemTrialBank
K_MAX = 6
def even(k): return [m for m in range(1 << k) if bin(m).count("1") % 2 == 0]
DESIGNS = {"k=2, full (4 drafts)": (2, list(range(4))), "k=3, half (4 drafts)": (3, even(3)), "k=3, full (8 drafts)": (3, list(range(8))),
           "k=4, half (8 drafts)": (4, even(4)), "k=5, half (16 drafts)": (5, even(5)), "k=6, half (32 drafts)": (6, even(6))}
OUT = S.OUT


def episode(beta, seed, M=12, warm=12, p_inf=0.7, noise_only=False, gamma=5.0, kappa=0.0):
    D, BASE, REL, RET, REG = S.D, S.BASE, S.REL, S.RET, S.REG
    rng = np.random.default_rng(seed)
    types = np.array([0] * M) if noise_only else np.array([1] * (M // 4) + [0] * (M - 2 * (M // 4)) + [-1] * (M // 4)); rng.shuffle(types)
    home = rng.integers(0, 2, M)
    rng5 = np.random.default_rng(seed + 4 * 10 ** 6)
    Zc = {int(m): list(kappa * types[m] * np.eye(8)[0] + rng5.normal(0, 1, 8)) for m in range(M)}
    rngS, rngP = np.random.default_rng(seed + 11 * 10 ** 6), np.random.default_rng(seed + 12 * 10 ** 6)
    rngB, rngR = np.random.default_rng(seed + 13 * 10 ** 6), np.random.default_rng(seed + 14 * 10 ** 6)
    banks = {n: MemTrialBank(Zc, {"MemTrial": dict(learner="auto")}, masks=ms) for n, (k, ms) in DESIGNS.items()}
    res = collections.defaultdict(list); opens = collections.defaultdict(list)
    for t in range(len(D)):
        score = (home == REG[t]).astype(float) + rngS.normal(0, 0.5, M); ret = [int(m) for m in np.argsort(-score)[:K_MAX]]
        preds = {}
        for m in ret:
            u, c = rngP.random(), int(rngP.integers(0, 4))
            preds[m] = int(np.argmax(RET[t])) if (types[m] == 1 and u < p_inf) else int(np.argmin(RET[t])) if (types[m] == -1 and u < p_inf) else c
        bases = [BASE[rngB.integers(len(BASE))] for _ in range(32)]
        ref_vec = np.mean([BASE[rngR.integers(len(BASE))] for _ in range(8)], 0)
        rel = REL[t]; refavg = float(S.U_batch(ref_vec[None, :], rel, gamma)[0])
        def act(sub, base):
            if not sub: return base
            tilt = np.mean([np.eye(4)[preds[m]] for m in sub], 0); return (1 - beta) * base + beta * tilt
        if t >= warm: res["reference (8 memory-free drafts)"].append(refavg)
        for n, (k, ms) in DESIGNS.items():
            ids = ret[:k]
            drafts = {mask: act([ids[j] for j in range(k) if (mask >> j) & 1], bases[i]) for i, mask in enumerate(ms)}
            Wd = np.array([drafts[mask] for mask in ms]); ud = S.U_batch(Wd, rel, gamma); umem = {mask: float(x) for mask, x in zip(ms, ud)}
            ens = Wd.mean(0); u_ens = float(S.U_batch(ens[None, :], rel, gamma)[0])
            grid = S.U_batch(S.GRID[:, None] * ref_vec[None, :] + (1 - S.GRID)[:, None] * ens[None, :], rel, gamma)
            b = banks[n]; v = b.decide(t, ids, umem, refavg, u_ens, grid)["MemTrial"]
            if t >= warm:
                res[n].append(v); opens[n].append(float(b.log["MemTrial"][-1][1]))
            b.matured(t, ids, umem, refavg, u_ens, grid)
    return {k: float(np.mean(v)) for k, v in res.items()}, {k: float(np.mean(v)) for k, v in opens.items()}


def _job(a):
    gname, n, s = a
    return episode(seed=s, gamma=S.GAMMA[gname], **S.REGIMES[n])


if __name__ == "__main__":
    mode = sys.argv[1]
    if mode in ("run", "prun"):
        gname, s0, s1 = sys.argv[2], int(sys.argv[3]), int(sys.argv[4]); t0 = time.time()
        jobs = [(gname, n, s) for n in S.REGIMES for s in range(s0, s1)]
        if mode == "prun":
            import multiprocessing as mp
            with mp.get_context("fork").Pool(4) as pool: outs = pool.map(_job, jobs, chunksize=2)
        else:
            outs = [_job(j) for j in jobs]
        R = collections.defaultdict(dict)
        for (g_, n, s), o in zip(jobs, outs): R[n][str(s)] = o
        json.dump(R, open(OUT / f"k_{gname}_{s0}_{s1}.json", "w")); print(gname, s0, s1, f"{time.time() - t0:.0f}s", flush=True)
    elif mode == "time":
        t0 = time.time(); r = episode(seed=20, gamma=5.0, beta=0.25); print(f"one episode {time.time() - t0:.2f}s"); print(r)
    elif mode == "table":
        s0, s1 = int(sys.argv[2]), int(sys.argv[3])
        ALL = collections.defaultdict(dict)
        for f in glob.glob(str(OUT / "k_*_*_*.json")):
            g = Path(f).stem.split("_")[1]
            for n, Sd in json.load(open(f)).items(): ALL[(g, n)].update({s: v for s, v in Sd.items() if s0 <= int(s) < s1})
        core = ["no influence", "noise only", "beta 0.1", "beta 0.25", "beta 0.5"]; regs = list(S.REGIMES)
        names = ["reference (8 memory-free drafts)"] + list(DESIGNS)
        L = [f"# k sensitivity, PlantedMem (seeds {s0}-{s1 - 1}; pp per month, mean over investors and seeds)", "",
             "| design | " + " | ".join(regs) + " | core avg | open share (core) |", "|---" * (len(regs) + 3) + "|"]
        for nm in names:
            cells, core_v, op = [], [], []
            for n in regs:
                v = [ALL[(g, n)][s][0][nm] for g in S.GAMMA for s in ALL[(g, n)]]
                cells.append(f"{100*np.mean(v):.3f}")
                if n in core:
                    core_v.append(np.mean(v))
                    if nm in DESIGNS: op.append(np.mean([ALL[(g, n)][s][1][nm] for g in S.GAMMA for s in ALL[(g, n)]]))
            per_seed = [np.mean([np.mean([ALL[(g, n)][s][0][nm] for g in S.GAMMA]) for n in core]) for s in sorted(ALL[("balanced", core[0])], key=int)]
            L.append(f"| {nm} | " + " | ".join(cells) + f" | {100*np.mean(core_v):.3f} ± {100*np.std(per_seed, ddof=1):.3f} | " + (f"{100*np.mean(op):.1f}%" if op else "") + " |")
        (OUT / "K_TABLE.md").write_text("\n".join(L)); print("\n".join(L))
