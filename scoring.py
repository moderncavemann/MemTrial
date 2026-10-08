"""Exact offline settlement and investor scoring of any weight vector (PortBench native rule, validated to <= 6e-6
against logged J). Investor p: Euclidean projection onto p's mandate, then U_p = J - (gamma_p / 2) * Var20.
Also: mandate violation of the unprojected weights (L1 move needed: excess risky + missing defensive share)."""
import sys, json, hashlib, atexit
from pathlib import Path
import numpy as np
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE / "portbench_prices"))
import rescore as RS
INVESTORS = list(RS.PROFILES)          # conservative, balanced, aggressive
_CACHE_P = HERE / "scoring_cache.json"
_C = json.load(open(_CACHE_P)) if _CACHE_P.exists() else {}
_dirty = [False]
@atexit.register
def _save():
    if _dirty[0]:
        tmp = _CACHE_P.with_suffix(".tmp"); json.dump(_C, open(tmp, "w")); tmp.replace(_CACHE_P); _dirty[0] = False
import signal
signal.signal(signal.SIGTERM, lambda *a: sys.exit(143))   # `timeout` -> SystemExit -> atexit caches are saved
def _key(w, d):
    s = d + "|" + ",".join(sorted(w)) + "|" + ",".join(f"{a}:{w[a]:.6f}" for a in sorted(w) if w[a] > 5e-7)
    return hashlib.sha1(s.encode()).hexdigest()
def score(w, d, universe=None):
    """w: weights dict (normalised here); universe: tradable assets of the date (defaults to keys of w)."""
    tot = sum(w.values()); w = {a: v / tot for a, v in w.items()}
    uni = sorted(universe or w)
    for a in uni: w.setdefault(a, 0.0)
    k = _key(w, d)
    if k in _C: return _C[k]
    prev = {a: 1 / len(uni) for a in uni}
    raw = RS.settle(w, prev, d); risky, dfn = RS.classes(w)
    out = dict(J=raw["J"], var20=raw["var20"], fee=raw["fee"], risky=risky, defn=dfn)
    for p, prof in RS.PROFILES.items():
        pw = RS.project(w, uni, prof); s = RS.settle(pw, prev, d)
        out[f"J_{p}"] = s["J"]; out[f"U_{p}"] = RS.utility(s, prof)
        out[f"viol_{p}"] = max(0.0, risky - prof["M"]) + max(0.0, prof["m"] - dfn)
    _C[k] = out; _dirty[0] = True
    return out
