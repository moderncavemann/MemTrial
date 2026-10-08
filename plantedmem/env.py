"""PlantedMem environment data (real prices and real LLM drafts; offline).

env_cache.pkl holds
  dates        : one decision per month, 2015-2026 (the first session of the month with 60 sessions of history and 20
                 sessions of outcome), each with the relative value path over the next 20 sessions of six asset-class
                 indices (equal-weighted over PortBench's assets of the class; a class without prices behaves like cash),
                 their 20-session returns, and the market regime (1 if the equity assets rose on average over the last
                 60 sessions);
  base_actions : asset-class weights of 2,828 logged gpt-4.1-mini PortBench allocations, the pool from which memory-free
                 drafts are resampled.
rebuild_dates() recomputes the dates from PortBench's prices (../portbench_prices) and returns the same values as the cache."""
import sys, collections, pickle
from pathlib import Path
import numpy as np
HERE = Path(__file__).resolve().parent
CLASSES = ["equities", "bonds", "cash", "commodities", "cryptocurrency", "real_estate"]   # PortBench class names
_CACHE = HERE / "env_cache.pkl"


def build():
    return pickle.load(open(_CACHE, "rb"))


def rebuild_dates():
    sys.path.insert(0, str(HERE.parent)); import scoring as SC
    RS = SC.RS
    members = collections.defaultdict(list)
    for a, c in RS.CMAP.items():
        if c in CLASSES and RS.PR.get(a): members[c].append(a)
    S = RS.SESS; dates = []; seen = set()
    for i, s in enumerate(S):
        key = (s.year, s.month)
        if key in seen or i < 61 or i + 20 >= len(S): continue
        seen.add(key); dates.append(i)
    D = []
    for i in dates:
        d = S[i]; fut = S[i + 1: i + 21]; past = S[i - 60: i + 1]
        rel = np.ones((len(CLASSES), 20))
        for k, c in enumerate(CLASSES):
            paths = []
            for a in members[c]:
                p = RS.PR[a]
                if d not in p or any(t not in p for t in fut): continue
                paths.append([p[t] / p[d] for t in fut])
            if paths: rel[k] = np.mean(paths, 0)
            else: rel[k] = rel[CLASSES.index("cash")] if k > 2 else 1.0   # unavailable class behaves like cash
        eq = [RS.PR[a] for a in members["equities"]]
        tr = [p[d] / p[past[0]] - 1 for p in eq if d in p and past[0] in p]
        D.append(dict(date=str(d), rel=rel, regime=1 if np.mean(tr) > 0 else 0, ret=rel[:, -1] - 1))
    return D


def utility(w, rel, gamma=5.0, fee=0.0015):
    w = w / w.sum(); prev = np.full(len(w), 1 / len(w)); nav = 1 - fee * np.abs(w - prev).sum()
    path = np.concatenate([[1.0], nav * (w @ rel)]); r = path[1:] / path[:-1] - 1
    return path[-1] - 1 - 0.5 * gamma * r.var() * 20
