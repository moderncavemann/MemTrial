"""PlantedMem summary of the never-trust ablation and of every Figure 4 variant, per regime group (offline).
Per seed: mean over the three investors and the regimes of a group; then mean and sd over the 100 test seeds (15-114)."""
import json, glob, collections
from pathlib import Path
import numpy as np
HERE = Path(__file__).resolve().parent; LAB = HERE.parent
CT = collections.defaultdict(dict)
for f in glob.glob(str(LAB / "memtrial/sens/ct_*_*_*.json")):
    g = Path(f).stem.split("_")[1]
    for n, S in json.load(open(f)).items():
        for s, v in S.items(): CT[(g, n)][s] = v[0]
for f in sorted(glob.glob(str(HERE / "nt_pm_*_*.json"))):
    R = json.load(open(f))
    for g in R:
        for n in R[g]:
            for s, v in R[g][n].items():
                CT[(g, n)].setdefault(s, {}).update({k: x for k, x in v[0].items() if k.startswith("NT|") or k.startswith("_open|NT")})
G3 = ("conservative", "balanced", "aggressive"); SEEDS = [str(s) for s in range(15, 115)]
GROUPS = {"no signal": ["no influence", "noise only"], "signal": ["beta 0.1", "beta 0.25", "beta 0.5"],
          "core": ["no influence", "noise only", "beta 0.1", "beta 0.25", "beta 0.5"],
          "many": ["many experiences"], "informative": ["many experiences, informative content"]}
def per_seed(key, regs): return np.array([np.mean([CT[(g, n)][s][key] for g in G3 for n in regs]) for s in SEEDS])
V = ["MemTrial", "MemTrial | identity learner", "MemTrial | content learner only", "MemTrial | F-test gate", "MemTrial | no gate",
     "NT|never trust", "MemTrial | closed->ensemble", "MemTrial | closed->reference", "MemTrial | uniform prior",
     "no-memory 8-draw ensemble (reference)", "W1N|MemTrial", "W1N|fallback to 1/N", "W1N|uniform prior", "NT|never trust, 1/N reference"]
out = {}
for k in V:
    out[k] = {}
    for gname, regs in GROUPS.items():
        x = 100 * per_seed(k, regs); out[k][gname] = [float(x.mean()), float(x.std(ddof=1)), x.tolist()]
    if "_open|" + k in CT[("balanced", "beta 0.25")]["15"]:
        out[k]["trust_core"] = float(100 * per_seed("_open|" + k, GROUPS["core"]).mean())
json.dump(out, open(HERE / "NT_PM_SUMMARY.json", "w"))
print(f"{'variant':42s}" + "".join(f"{g:>16s}" for g in GROUPS))
for k in V: print(f"{k:42s}" + "".join(f"{out[k][g][0]:9.3f}±{out[k][g][1]:.3f}" for g in GROUPS))
mt = {g: np.array(out["MemTrial"][g][2]) for g in GROUPS}
for k in ("NT|never trust", "MemTrial | no gate", "MemTrial | closed->reference"):
    d = {g: np.array(out[k][g][2]) - mt[g] for g in GROUPS}
    print("paired diff vs MemTrial", k, {g: (round(float(d[g].mean()), 4), "identical" if np.all(d[g] == 0) else "") for g in GROUPS})
