"""Hyperparameter sensitivity and weak-reference robustness of the frozen MemGate (2026-10-04). Offline, no LLM calls.
memgate.py is NOT modified (frozen; hash in MEMGATE_FROZEN.json). SensBank re-implements MemGateBank.decide with
per-variant (alpha, a0, lam0, min_scores); the original variant names are recomputed too and must equal the stored
pb_all / results values. Two extra worlds read the caller's locals (analysis code only):
  pb16 (PortBench): a second bank on all 16 subsets, 16-draft average in the closed state -> 'S16|...'
  ct1n (PlantedMem): a second anchored allocation with the 1/N portfolio as the (weak) reference -> 'W1N|...'
usage: python3 sens.py pb <cfg> <investor>        -> sens/pb_<cfg>_<investor>.json
       python3 sens.py ct <investor> <s0> <s1>    -> sens/ct_<investor>_<s0>_<s1>.json"""
import sys, os, json, math, glob, time, collections
from pathlib import Path
import numpy as np
HERE = Path(__file__).resolve().parent; LAB = HERE.parent; OUT = HERE / "sens"; OUT.mkdir(exist_ok=True)
sys.path.insert(0, str(HERE))
import memgate as MGm
from memgate import MemGateBank, ftest, ftrl_q, t_sf, GRID

SENS_V = {"S|default": dict(learner="auto")}
for a in (0.01, 0.02, 0.1, 0.2): SENS_V[f"S|alpha={a}"] = dict(learner="auto", alpha=a)
for a0 in (0.5, 0.75, 0.9, 0.95):
    for l0 in (1.0, 4.0, 16.0):
        if (a0, l0) != (0.9, 4.0): SENS_V[f"S|a0={a0},lam0={l0}"] = dict(learner="auto", a0=a0, lam0=l0)
for ms in (5, 20): SENS_V[f"S|minscores={ms}"] = dict(learner="auto", minsc=ms)
W1N_V = {"W1N|MemGate": dict(learner="auto", closed="ftrl1n"), "W1N|fallback to 1/N": dict(learner="auto", closed="ref1n"),
         "W1N|uniform prior": dict(learner="auto", closed="uniform1n")}
S16_V = {"S16|MemGate": dict(learner="auto"), "S16|no gate": dict(learner="auto", gate="none")}


class SensBank(MemGateBank):
    def __init__(self, Z, variants, masks=None, a0=0.9, lam0=4.0, extra=None):
        super().__init__(Z, variants, masks, a0, lam0)
        self.extra = extra; self.sumGN = np.zeros(len(GRID)); self.nN = 0; self.sigN = 0.0; self._pend = None
        self.sub = SensBank(Z, S16_V, masks=list(range(16))) if extra == "pb16" else None

    def _p(self, m, ms):
        sc = [x for x in self.G[m].scores if np.isfinite(x)]
        if len(sc) < ms: return 1.0
        sd = np.std(sc, ddof=1)
        if sd == 0: return 0.0 if np.mean(sc) > 0 else 1.0
        return t_sf(np.mean(sc) / (sd / math.sqrt(len(sc))), len(sc) - 1)

    def decide(self, t, ids, U, u_ref, u_ens, grid):
        if self.extra == "ct1n":
            fr = sys._getframe(1).f_locals; ew, ens_vec, rel, gamma = fr["ew"], fr["ens_vec"], fr["rel"], fr["gamma"]
            import suite as V5
            gridN = V5.U_batch(GRID[:, None] * ew[None, :] + (1 - GRID)[:, None] * ens_vec[None, :], rel, gamma)
            self._pend = (float(fr["u_ew"]), gridN)
        if self.extra == "pb16":
            fr = sys._getframe(1).f_locals; u, ref, ens16, d, p, D = fr["u"], fr["ref"], fr["ens16"], fr["d"], fr["p"], fr["D"]
            import monthly_policies as MP
            self._pend16 = ({m: u[str(m)] for m in range(16)}, u["ENS16"], MP.ugrid(D.pack(np.array([ref, ens16]), d, p)))
        st = {}
        for m, L in self.L.items():
            L.fit(self.records); v = [L.predict(h) for h in ids]; self.G[m].predicted(t, v)
            best = max(self.masks, key=lambda k: (sum(v[j] for j in range(len(ids)) if (k >> j) & 1), -bin(k).count("1")))
            st[m] = (best, sum(v[j] for j in range(len(ids)) if (best >> j) & 1))
        pf = None; qc = {}
        def q_of(a, l, w="main"):
            if (a, l, w) not in qc:
                qc[(a, l, w)] = ftrl_q(self.sumG, self.n, self.sig, a, l) if w == "main" else ftrl_q(self.sumGN, self.nN, self.sigN, a, l)
            return qc[(a, l, w)]
        out = {}
        for name, cfg in self.V.items():
            ms = cfg.get("minsc", MGm.MIN_SCORES); lm = cfg.get("learner", "content")
            if lm == "auto":
                sc = {m: np.mean(self.G[m].scores) if len(self.G[m].scores) >= ms else -np.inf for m in ("identity", "content")}
                lm = "content" if sc["content"] > sc["identity"] else "identity"
            best, gain = st[lm]; kind = cfg.get("gate", "prequential")
            if kind == "prequential": p = self._p(lm, ms)
            elif kind == "ftest":
                if pf is None: pf = ftest(self.records)
                p = pf
            else: p = 0.0
            opened = p < cfg.get("alpha", 0.05) and gain > 0
            cl = cfg.get("closed", "ftrl"); a0 = cfg.get("a0", self.a0); l0 = cfg.get("lam0", self.lam0)
            if opened: val = U[best]
            elif cl == "ens": val = u_ens
            elif cl == "ref": val = u_ref
            elif cl == "uniform": val = float(np.interp(q_of(0.5, l0), GRID, grid))
            elif cl == "ref1n": val = self._pend[0]
            elif cl == "ftrl1n": val = float(np.interp(q_of(a0, l0, "1n"), GRID, self._pend[1]))
            elif cl == "uniform1n": val = float(np.interp(q_of(0.5, l0, "1n"), GRID, self._pend[1]))
            else: val = float(np.interp(q_of(a0, l0), GRID, grid))
            out[name] = val; self.log[name].append((t, opened, p))
        if self.sub is not None:
            U16, ue16, g16 = self._pend16; out.update(self.sub.decide(t, ids, U16, u_ref, ue16, g16))
            for k in S16_V: self.log[k] = self.sub.log[k]
        return out

    def matured(self, t, ids, U, u_ref, u_ens, grid):
        super().matured(t, ids, U, u_ref, u_ens, grid)
        if self.extra == "ct1n":
            u_ew, gridN = self._pend; self.sumGN += gridN; self.nN += 1; self.sigN += abs(u_ew - u_ens) / 2.0
        if self.sub is not None:
            U16, ue16, g16 = self._pend16; self.sub.matured(t, ids, U16, u_ref, ue16, g16)


def run_pb(cfg, p):
    sys.path.insert(0, str(LAB / "portbench")); import pb_all as PA
    PA.MemGateBank = lambda Z, V, masks=None: SensBank(Z, {**PA.VARIANTS, **SENS_V}, masks=masks, extra="pb16")
    rows, ids, dates, split = PA.ML.load_monthly(); inp = json.load(open(PA.HERE / "data/inputs.json"))
    D = PA.ML.Data(rows, ids, dates, split, cache=PA.ML.load_cache())
    Z = PA.content_features(D, inp); t0 = time.time()
    res, gl = PA.run(D, cfg, p, Z)
    keep = {k: v for k, v in res.items() if k.startswith(("S|", "S16|", "MemGate")) or k == "1/N"}
    json.dump({"res": keep, "split": split}, open(OUT / f"pb_{cfg}_{p}.json", "w")); print(cfg, p, f"{time.time() - t0:.0f}s")


def run_ct(g, s0, s1):
    sys.path.insert(0, str(LAB / "plantedmem")); import suite as V5
    V5.MemGateBank = lambda Z, V, masks=None: SensBank(Z, {**V5.MG_VARIANTS, **SENS_V, **W1N_V}, masks=masks, extra="ct1n")
    import multiprocessing as mp
    jobs = [(g, n, s) for n in V5.REGIMES for s in range(s0, s1)]; t0 = time.time()
    with mp.get_context("fork").Pool(4) as pool: outs = pool.map(V5._job, jobs, chunksize=5)
    R = collections.defaultdict(dict)
    for (g_, n, s), o in zip(jobs, outs): R[n][str(s)] = [{k: v for k, v in o[0].items() if "|" in k or k.startswith(("MemGate", "1/N", "uniform", "no-memory", "counterfactual"))}, o[1]]
    json.dump(R, open(OUT / f"ct_{g}_{s0}_{s1}.json", "w")); print(g, s0, s1, f"{time.time() - t0:.0f}s")



if __name__ == "__main__":
    mode = sys.argv[1]
    if mode == "pb": run_pb(sys.argv[2], sys.argv[3])
    elif mode == "ct": run_ct(sys.argv[2], int(sys.argv[3]), int(sys.argv[4]))
