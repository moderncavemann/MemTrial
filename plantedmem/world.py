"""PlantedMem world: the market paths and the pool of logged LLM drafts of env.py, restricted to four asset classes
(equities, bonds, cash, commodities), and the half-fraction design over the four retrieved experiences."""
import sys
from pathlib import Path
import numpy as np
HERE = Path(__file__).resolve().parent; sys.path.insert(0, str(HERE))
import env
E = env.build(); D = E["dates"]; K4 = [0, 1, 2, 3]           # equities, bonds, cash, commodities
BASE = E["base_actions"][:, K4]; BASE = BASE[BASE.sum(1) > 0]; BASE = BASE / BASE.sum(1, keepdims=True)
REL = [d["rel"][K4] for d in D]; RET = np.array([d["ret"][K4] for d in D]); REG = np.array([d["regime"] for d in D])
HALF = [0, 3, 5, 6, 9, 10, 12, 15]                           # 2^(4-1) half fraction (even parity): main effects unaliased with pairs
