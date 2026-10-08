"""Shared helpers for the sd computations (sd_*.py). Analysis only: reads published result files, never writes to them."""
import sys, json
from pathlib import Path
import numpy as np
HERE = Path(__file__).resolve().parent; LAB = HERE.parent
for p in (LAB / "memtrial", LAB / "portbench", LAB / "investorbench", LAB / "classalloc_and_robustness"):
    sys.path.insert(0, str(p))
G3 = ("conservative", "balanced", "aggressive")


def ms(x):
    x = np.asarray(x, float).ravel()
    return [float(x.mean()), float(x.std(ddof=1)) if len(x) > 1 else 0.0]


def spearman(a, b):
    ra = np.argsort(np.argsort(a)); rb = np.argsort(np.argsort(b)); return float(np.corrcoef(ra, rb)[0, 1])


def save(name, obj):
    (HERE / f"SD_{name}.json").write_text(json.dumps(obj, indent=1)); print(f"wrote SD_{name}.json")
