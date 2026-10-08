"""Shared definitions of the PortBench evaluation (pb_all.py): the 16 subsets (masks) of the four retrieved experiences
and their half fraction, the memory configurations that Hedge mixes, Banzhaf contributions, Hedge weights, and the exact
utility of blends of the reference and the draft average on a grid (MemTrial's anchored action)."""
import sys, math
from pathlib import Path
import numpy as np
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import monthly_lib as ML                                           # also puts the shared modules on the import path
import mixture as CM
MASKS = [str(m) for m in range(16)]
HALF = ["0", "3", "5", "6", "9", "10", "12", "15"]
FAMILIES = ["0", "15", "FinMem", "MemRL", "M2_original"]          # no memory, all four retrieved, the memory providers (Hedge)


def banzhaf(T, j): return float(np.mean([T[m] for m in MASKS if (int(m) >> j) & 1]) - np.mean([T[m] for m in MASKS if not (int(m) >> j) & 1]))


def hedge(past, prior, lam0=4.0):
    if not past: return prior
    U = np.array(past); n = len(U); sig = float(np.mean(U.std(1))) + 1e-9; lam = lam0 * sig / math.sqrt(n)
    z = np.log(prior) + U.mean(0) / lam; q = np.exp(z - z.max()); return q / q.sum()


GRID = np.linspace(0.0, 1.0, 201)


def ugrid(pack):
    """exact utility of q*ref + (1-q)*ens for q on GRID (pack = two-row expert pack)."""
    X, G, prev, gam, _ = pack
    Q = np.stack([GRID, 1 - GRID], 1)
    return CM.U_batch(Q, X, G, prev, gam)
