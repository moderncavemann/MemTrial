"""How many retrieved experiences (k)? InvestorBench sensitivity of MemGate with the existing drafts (python3 ib_k.py) ->
turnover/K_IB.json. Fee on traded amounts as in ib_turnover.py. The logged drafts cover every subset of the 4 retrieved
experiences, so k = 2, 3, 4 can be evaluated without new LLM calls: MemGate uses the first k retrieved experiences
(by similarity) and a designed set of their subsets. memgate.py unmodified."""
import json, numpy as np
import ib_turnover as T
from memgate import MemGateBank, GRID, ftrl_q
def even(k): return [m for m in range(1 << k) if bin(m).count("1") % 2 == 0]
DESIGNS = {"k=2, full (4 drafts)": (2, list(range(4))), "k=3, half (4 drafts)": (3, even(3)), "k=3, full (8 drafts)": (3, list(range(8))),
           "k=4, half (8 drafts)": (4, even(4)), "k=4, full (16 drafts)": (4, list(range(16)))}


def run(k, masks, C):
    W, top4, seeds, dates, ret, Z = C["W"], C["top4"], C["seeds"], C["dates"], C["ret"], C["Z"]
    res = np.zeros((len(dates), 3, len(seeds))); op = np.zeros_like(res)
    for pi, (p, (M, mfl, gam)) in enumerate(T.INV.items()):
        ew = T.project(np.full(5, 0.2), M, mfl)
        for si, s in enumerate(seeds):
            bank = MemGateBank(Z, {"MemGate": dict(learner="auto")}, masks=masks); h = ew.copy()
            for di, t in enumerate(dates):
                w = {kk: T.project(v, M, mfl) for kk, v in W[(t, s)].items()}; r = ret[t]; ids = top4[t][:k]
                ens = np.mean([w[f"m{m}"] for m in masks], 0)
                U = {m: float(T.util(w[f"m{m}"], r, gam, h)[0]) for m in masks}
                u_ref = float(T.util(ew, r, gam, h)[0]); u_ens = float(T.util(ens, r, gam, h)[0])
                grid = T.util(GRID[:, None] * ew[None, :] + (1 - GRID)[:, None] * ens[None, :], r, gam, h)
                q = ftrl_q(bank.sumG, bank.n, bank.sig, 0.9, 4.0)
                out = bank.decide(t, ids, U, u_ref, u_ens, grid)["MemGate"]; o = bool(bank.log["MemGate"][-1][1])
                if o:
                    sc = {m: np.mean(bank.G[m].scores) if len(bank.G[m].scores) >= T.MG.MIN_SCORES else -np.inf for m in ("identity", "content")}
                    lm = "content" if sc["content"] > sc["identity"] else "identity"; v = [bank.L[lm].predict(x) for x in ids]
                    best = max(masks, key=lambda mm: (sum(v[j] for j in range(k) if (mm >> j) & 1), -bin(mm).count("1"))); X = w[f"m{best}"]
                else:
                    X = q * ew + (1 - q) * ens
                if abs(float(T.util(X, r, gam, h)[0]) - out) > 1e-10: raise RuntimeError("deployed portfolio mismatch")
                bank.matured(t, ids, U, u_ref, u_ens, grid); res[di, pi, si] = out; op[di, pi, si] = o; h = T.drift(X, r)
    return res, op


if __name__ == "__main__":
    C = T.load(); out = {}
    J = json.load(open(T.HERE / "ib_all_turnover.json")); n1 = 1e4 * np.array(J["res"]["1/N"], float).mean()
    for name, (k, masks) in DESIGNS.items():
        res, op = run(k, masks, C); ps = 1e4 * res.mean((0, 1))
        out[name] = {"mean_bp": float(ps.mean()), "sd_bp": float(ps.std(ddof=1)), "open_share": float(op.mean()), "drafts": len(masks)}
        print(f"{name:24s} {ps.mean():6.2f} ± {ps.std(ddof=1):.2f} bp/day   open {100*op.mean():5.2f}%", flush=True)
    out["1/N"] = {"mean_bp": float(n1)}
    (T.OUT / "K_IB.json").write_text(json.dumps(out, indent=1)); print("1/N", round(n1, 2))
