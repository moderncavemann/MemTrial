"""PlantedMem: sd over test seeds for every PlantedMem number reported without one.
python3 sd_pm.py -> SD_pm.json. Seeds 15-114 (k tables: 15-64), mean over investors (and the five core regimes for 'core')."""
import glob, collections, json
from pathlib import Path
import numpy as np
from sd_common import *
from paper_tables import ct_load, CORE, CT
INF = "many experiences, informative content"; MANY = "many experiences"; REGS = CORE + [MANY, INF]
ALL = ct_load()                                                   # suite5, seeds >= 15
SENS = collections.defaultdict(dict)
for f in glob.glob(str(LAB / "memtrial/sens/ct_*_*_*.json")):
    g = Path(f).stem.split("_")[1]
    for n, S in json.load(open(f)).items(): SENS[(g, n)].update({s: v for s, v in S.items() if 15 <= int(s) < 115})
K = collections.defaultdict(dict)
for pat in ("k_", "k16_"):
    for f in glob.glob(str(LAB / f"plantedmem/results/{pat}*_*_*.json")):
        g = Path(f).stem.split("_")[1]
        for n, S in json.load(open(f)).items():
            for s, v in S.items():
                if 15 <= int(s) < 65:
                    e = K[(g, n)].setdefault(s, [{}, {}]); e[0].update(v[0]); e[1].update(v[1] if len(v) > 1 and isinstance(v[1], dict) else {})


def per_seed(store, key, regs, idx=0, scale=100.0):
    seeds = sorted(store[("balanced", regs[0])], key=int)
    return np.array([np.mean([store[(g, n)][s][idx][key] for g in G3 for n in regs]) for s in seeds]) * scale


out = {"n_seeds_core": len(ALL[("balanced", CORE[0])]), "n_seeds_k": len(K[("balanced", CORE[0])])}
# ---- main table derived rows (core regimes)
EXP = ["FinMem", "MemRL", "Reflexion", "ExpeL"]
mt = per_seed(ALL, "MemTrial", CORE); best = max(EXP, key=lambda k: per_seed(ALL, CT[k], CORE).mean())
out["main"] = {"best_experience_agent": best, "improv": ms(mt - per_seed(ALL, CT[best], CORE)), "minus_1N": ms(mt - per_seed(ALL, "1/N", CORE)),
               "trust_pct": ms(per_seed(ALL, "_open|MemTrial", CORE))}
# ---- ablation / trust-sensitivity trust rates (core and informative)
ABL = ["MemTrial", "MemTrial | identity learner", "MemTrial | content learner only", "MemTrial | F-test gate", "MemTrial | no gate",
       "MemTrial | closed->ensemble", "MemTrial | closed->reference", "MemTrial | uniform prior"]
out["trust_ablation"] = {n: {"core": ms(per_seed(ALL, "_open|" + n, CORE)), "informative": ms(per_seed(ALL, "_open|" + n, [INF]))} for n in ABL}
SENSN = ["S|default", "S|alpha=0.01", "S|alpha=0.02", "S|alpha=0.1", "S|alpha=0.2", "S|minscores=5", "S|minscores=20"]
out["trust_sens"] = {n: {"core": ms(per_seed(SENS, "_open|" + n, CORE)), "informative": ms(per_seed(SENS, "_open|" + n, [INF]))} for n in SENSN}
out["util_sens"] = {n: {"core": ms(per_seed(SENS, n, CORE)), "informative": ms(per_seed(SENS, n, [INF]))} for n in SENSN}
# ---- anchor table (core)
AN = [k for k in next(iter(SENS.values()))["15"][0] if k.startswith("S|a0=")]
out["anchor_core"] = {n: ms(per_seed(SENS, n, CORE)) for n in AN}
out["anchor_core"]["default (MemTrial)"] = ms(per_seed(SENS, "MemTrial", CORE))
# ---- trust rate by regime
TR = {"MemTrial (forward test)": "_open|MemTrial", "Per-experience learner only": "_open|MemTrial | identity learner", "In-sample F-test": "_open|MemTrial | F-test gate"}
out["trust_by_regime"] = {lab: {n: ms(per_seed(ALL, key, [n])) for n in REGS} for lab, key in TR.items()}
# ---- utility by regime
UR = {"Reference (memory-free average)": "no-memory 8-draw ensemble (reference)", "Outcome credit (MemRL)": "outcome credit top-2 (MemRL/FinMem-style)",
      "Counterfactual selection": "counterfactual credit argmax top-2", "MemTrial w/o trust test": "MemTrial | no gate", "MemTrial": "MemTrial"}
out["util_by_regime"] = {lab: {n: ms(per_seed(ALL, key, [n])) for n in REGS} for lab, key in UR.items()}
# ---- weak reference (1/N as the reference)
WR = {"1/N": "1/N", "Draft averaging": "uniform aggregation of the 8 members", "MemTrial, fixed fallback to 1/N": "W1N|fallback to 1/N",
      "MemTrial, anchored (a0=0.9)": "W1N|MemTrial", "MemTrial, uniform prior (a0=0.5)": "W1N|uniform prior"}
out["weakref_by_regime"] = {lab: {**{n: ms(per_seed(SENS, key, [n])) for n in REGS}, "core": ms(per_seed(SENS, key, CORE))} for lab, key in WR.items()}
# ---- k and design (seeds 15-64)
KD = ["reference (8 memory-free drafts)", "k=2, full (4 drafts)", "k=3, half (4 drafts)", "k=3, full (8 drafts)", "k=4, half (8 drafts)",
      "k=4, full (16 drafts)", "k=5, half (16 drafts)", "k=6, half (32 drafts)"]
out["k_by_regime"] = {d: {**{n: ms(per_seed(K, d, [n])) for n in REGS}, "core": ms(per_seed(K, d, CORE))} for d in KD}
out["k_trust"] = {d: {"core": ms(per_seed(K, d, CORE, idx=1)), "informative": ms(per_seed(K, d, [INF], idx=1))} for d in KD if not d.startswith("reference")}
save("pm", out)
print(json.dumps(out["main"]), json.dumps(out["trust_by_regime"]["MemTrial (forward test)"]))
print("k=4 half core util", out["k_by_regime"]["k=4, half (8 drafts)"]["core"], "trust", out["k_trust"]["k=4, half (8 drafts)"])
print("weakref core", {k: v["core"] for k, v in out["weakref_by_regime"].items()})
