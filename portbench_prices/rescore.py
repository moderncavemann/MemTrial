"""Offline re-scorer of PortBench decisions (no network, no model calls).
settle(): PortBench's settlement and rebalancing:
  fee = 15 bps x sum|target - previous| (|delta| < 1e-6 ignored), charged once on NAV;
  target normalised; then buy-and-hold over the next 20 BIL sessions.
project(): Euclidean projection onto an investor mandate (PortBench's investor profiles):
  long-only simplex over the date's universe, sum(bonds+cash) >= m, sum(equities+crypto+real_estate) <= M.
  Dykstra's algorithm over the three convex sets."""
import pickle, json
from pathlib import Path
import numpy as np
HERE = Path(__file__).resolve().parent
D = pickle.load(open(HERE / "prices.pkl", "rb"))
SESS = [s.date() for s in D["sessions"]]
SIDX = {d: i for i, d in enumerate(SESS)}
PR = {a: {ts.date(): float(v) for ts, v in s.items()} for a, s in D["prices"].items()}
CMAP = D["class_map"]
FEE = 0.0015
PROFILES = {  # PortBench's investor profiles
    "conservative": dict(M=0.40, m=0.40, gamma=10.0),
    "balanced":     dict(M=0.65, m=0.20, gamma=5.0),
    "aggressive":   dict(M=0.90, m=0.05, gamma=2.0),
}
RISKY = {"equities", "cryptocurrency", "real_estate"}   # PortBench's risky classes
DEF = {"bonds", "cash"}                                 # PortBench's defensive classes

def last_price(a, d):
    p = PR[a]
    if d in p: return p[d]
    i = SIDX[d]
    while i > 0:
        i -= 1
        if SESS[i] in p: return p[SESS[i]]
    raise KeyError((a, d))

def settle(target, prev, d, horizon=20):
    d = d if not isinstance(d, str) else __import__("datetime").date.fromisoformat(d)
    pos = SIDX[d]; fut = SESS[pos + 1: pos + 1 + horizon]
    assert len(fut) == horizon
    turnover = sum(abs(target.get(a, 0) - prev.get(a, 0)) for a in set(target) | set(prev)
                   if abs(target.get(a, 0) - prev.get(a, 0)) >= 1e-6)
    tot = sum(target.values()); w = {a: v / tot for a, v in target.items() if abs(v / tot) > 1e-12}
    nav = 1.0 - FEE * turnover
    p0 = {a: last_price(a, d) for a in w}
    path = [1.0]
    for t in fut:
        val = 0.0
        for a, wa in w.items():
            if t not in PR[a]: raise ValueError(f"missing realized observation {a} {t}")
            val += wa * PR[a][t] / p0[a]
        path.append(nav * val)
    path = np.array(path)
    r = path[1:] / path[:-1] - 1
    peak = np.maximum.accumulate(path); mdd = float((path / peak - 1).min())
    return {"J": float(path[-1] - 1), "fee": FEE * turnover, "turnover": turnover,
            "var20": float(r.var(ddof=0) * horizon), "mdd": mdd}

def cls(a):
    """Strict: every traded asset must have a PortBench class (no silent 'unclassified')."""
    c = CMAP.get(a)
    if c is None: raise KeyError(f"asset without class: {a}")
    return c

def classes(w):
    risky = sum(v for a, v in w.items() if cls(a) in RISKY)
    dfn = sum(v for a, v in w.items() if cls(a) in DEF)
    return risky, dfn

def _simplex(y):
    u = np.sort(y)[::-1]; css = np.cumsum(u)
    rho = np.nonzero(u * np.arange(1, len(y) + 1) > (css - 1))[0][-1]
    theta = (css[rho] - 1) / (rho + 1.0)
    return np.maximum(y - theta, 0)

def project(w, universe, prof, iters=3000, tol=1e-12):
    assets = sorted(universe)
    y = np.array([w.get(a, 0.0) for a in assets])
    aB = np.array([1.0 if cls(a) in DEF else 0.0 for a in assets])
    aR = np.array([1.0 if cls(a) in RISKY else 0.0 for a in assets])
    m, M = prof["m"], prof["M"]
    if aB @ y >= m - 1e-12 and aR @ y <= M + 1e-12 and abs(y.sum() - 1) < 1e-9 and (y >= -1e-12).all():
        return {a: float(v) for a, v in zip(assets, y)}
    x = y.copy(); p = [np.zeros_like(y) for _ in range(3)]
    def P0(z): return _simplex(z)
    def P1(z): s = aB @ z; return z if s >= m else z + (m - s) / aB.sum() * aB
    def P2(z): s = aR @ z; return z if s <= M else z - (s - M) / aR.sum() * aR
    projs = [P0, P1, P2]
    for _ in range(iters):
        x_old = x.copy()
        for k, P in enumerate(projs):
            z = P(x + p[k]); p[k] = x + p[k] - z; x = z
        if np.abs(x - x_old).max() < tol: break
    x = _simplex(x)  # final feasibility clean-up on the simplex
    return {a: float(v) for a, v in zip(assets, x) if v > 1e-12}

def utility(res, prof):
    return res["J"] - 0.5 * prof["gamma"] * res["var20"]
