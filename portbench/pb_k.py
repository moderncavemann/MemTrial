"""How many retrieved experiences (k)? PortBench sensitivity of MemGate with the existing drafts (python3 pb_k.py <cfg>)
-> pb_k_<cfg>.json. The logged drafts cover every subset of the 4 retrieved experiences, so k = 2, 3, 4 are evaluated without
new LLM calls: MemGate uses the first k retrieved experiences (by similarity) and a designed set of their subsets.
Everything else as pb_all.py (frozen pipeline untouched; memgate.py unmodified)."""
import sys, os, json, collections
from pathlib import Path
import numpy as np
HERE = Path(__file__).resolve().parent; sys.path.insert(0, str(HERE)); sys.path.insert(0, str(HERE.parent / "memtrial"))
import monthly_lib as ML, monthly_policies as MP, pb_all as PA
from memgate import MemGateBank
def even(k): return [m for m in range(1 << k) if bin(m).count("1") % 2 == 0]
DESIGNS = {"k=2, full (4 drafts)": (2, list(range(4))), "k=3, half (4 drafts)": (3, even(3)), "k=3, full (8 drafts)": (3, list(range(8))),
           "k=4, half (8 drafts)": (4, even(4)), "k=4, full (16 drafts)": (4, list(range(16)))}


def run(D, cfg, p, Z):
    key = (cfg, p); MASKS = MP.MASKS
    valid = [d for d in D.dates if d in D.proj[key] and all(any(m in D.proj[key][d][r] for r in D.proj[key][d]) for m in MASKS)]
    res = {n: collections.defaultdict(list) for n in DESIGNS}; opens = {n: {} for n in DESIGNS}
    for r in (0, 1, 2):
        banks = {n: MemGateBank(Z, {"MemGate": dict(learner="auto")}, masks=ms) for n, (k, ms) in DESIGNS.items()}
        for d in valid:
            W = D.proj[key][d]
            def X(arm):
                if arm in W.get(r, {}): return W[r][arm]
                for rr in sorted(W):
                    if arm in W[rr]: return W[rr][arm]
            Xs = {m: X(m) for m in MASKS}; ref = D.ref[(d, p, "1/N")]
            u = dict(zip(list(MASKS) + ["1/N"], D.U(np.array([Xs[m] for m in MASKS] + [ref]), d, p)))
            for n, (k, ms) in DESIGNS.items():
                ens = np.mean([Xs[str(m)] for m in ms], 0); u_ens = float(D.U(np.array([ens]), d, p)[0])
                grid = MP.ugrid(D.pack(np.array([ref, ens]), d, p)); U = {m: float(u[str(m)]) for m in ms}; ids = D.ids[d][:k]
                v = banks[n].decide(d, ids, U, float(u["1/N"]), u_ens, grid)["MemGate"]
                res[n][d].append(float(v)); opens[n].setdefault(d, []).append(float(banks[n].log["MemGate"][-1][1]))
                banks[n].matured(d, ids, U, float(u["1/N"]), u_ens, grid)
    return res, opens


if __name__ == "__main__":
    cfg = sys.argv[1]
    rows, ids, dates, split = ML.load_monthly(); inp = json.load(open(HERE / "data/inputs.json"))
    D = ML.Data(rows, ids, dates, split, cache=ML.load_cache())
    Z = PA.content_features(D, inp)
    out = {}
    for p in ("conservative", "balanced", "aggressive"):
        res, opens = run(D, cfg, p, Z); out[p] = {"res": {n: dict(v) for n, v in res.items()}, "open": opens}
    json.dump({"out": out, "split": split}, open(HERE / f"pb_k_{cfg}.json", "w"))
    for n in DESIGNS:
        per = np.mean([[np.mean([out[p]["res"][n][d][w] for d in out[p]["res"][n] if d >= split]) for w in range(3)] for p in out], 0)
        op = np.mean([x for p in out for d, v in out[p]["open"][n].items() if d >= split for x in v])
        print(f"{cfg} {n:24s} {100*per.mean():.3f} ± {100*per.std(ddof=1):.3f} pp/month   open(test) {100*op:.1f}%", flush=True)
