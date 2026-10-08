"""Share of decisions on which MemGate/CAVEAT trusts its learned values, for every sensitivity variant (analysis only;
python3 trust_rates.py) -> TRUST_RATES.json. PortBench: the SensBank variants of sens.py replayed with the same inputs as
pb_all.py, test months, pooled over configurations, investors and seeds. InvestorBench: ../investorbench/turnover
(fee on traded amounts). Controlled: sens/ct_*.json, test seeds 15-114, five core regimes and the informative regime."""
import sys, os, json, glob, collections
from pathlib import Path
import numpy as np
HERE = Path(__file__).resolve().parent; LAB = HERE.parent
sys.path.insert(0, str(HERE)); sys.path.insert(0, str(LAB / "portbench"))
import sens as SE, monthly_lib as ML, monthly_policies as MP, pb_all as PA
HALF = [int(m) for m in MP.HALF]
V = {**PA.VARIANTS, **SE.SENS_V}
out = {"PortBench": {}, "InvestorBench": {}, "Controlled core": {}, "Controlled informative": {}}
rows, ids, dates, split = ML.load_monthly(); inp = json.load(open(LAB / "portbench" / "data/inputs.json"))
D = ML.Data(rows, ids, dates, split, cache=ML.load_cache()); Z = PA.content_features(D, inp)
acc = collections.defaultdict(list)
for cfg in ("full-price", "raw-price"):
    for p in ("conservative", "balanced", "aggressive"):
        key = (cfg, p)
        valid = [d for d in D.dates if d in D.proj[key] and all(any(m in D.proj[key][d][r] for r in D.proj[key][d]) for m in MP.MASKS)]
        for r in (0, 1, 2):
            bank = SE.SensBank(Z, V, masks=HALF)
            for d in valid:
                W = D.proj[key][d]
                def X(arm):
                    if arm in W.get(r, {}): return W[r][arm]
                    for rr in sorted(W):
                        if arm in W[rr]: return W[rr][arm]
                Xs = {m: X(m) for m in MP.MASKS}; ref = D.ref[(d, p, "1/N")]; ens8 = np.mean([Xs[m] for m in MP.HALF], 0)
                u = dict(zip(list(MP.MASKS) + ["1/N", "ENS8"], D.U(np.array([Xs[m] for m in MP.MASKS] + [ref, ens8]), d, p)))
                Uh = {m: float(u[str(m)]) for m in HALF}; grid = MP.ugrid(D.pack(np.array([ref, ens8]), d, p))
                bank.decide(d, D.ids[d], Uh, float(u["1/N"]), float(u["ENS8"]), grid)
                if d >= split:
                    for n in V: acc[n].append(float(bank.log[n][-1][1]))
                bank.matured(d, D.ids[d], Uh, float(u["1/N"]), float(u["ENS8"]), grid)
out["PortBench"] = {n: float(np.mean(v)) for n, v in acc.items()}
S = json.load(open(LAB / "investorbench" / "turnover" / "SUMMARY.json"))["open_share"]
out["InvestorBench"] = S
ALL = collections.defaultdict(dict)
for f in glob.glob(str(HERE / "sens" / "ct_*_*_*.json")):
    g = Path(f).stem.split("_")[1]
    for n, Sd in json.load(open(f)).items(): ALL[(g, n)].update({s: v for s, v in Sd.items() if 15 <= int(s) < 115})
core = ["no influence", "noise only", "beta 0.1", "beta 0.25", "beta 0.5"]
names = [k[6:] for k in next(iter(ALL.values()))[next(iter(next(iter(ALL.values()))))][0] if k.startswith("_open|")]
for n in names:
    out["Controlled core"][n] = float(np.mean([ALL[(g, rg)][s][0]["_open|" + n] for g in ("conservative", "balanced", "aggressive") for rg in core for s in ALL[(g, rg)]]))
    rg = "many experiences, informative content"
    out["Controlled informative"][n] = float(np.mean([ALL[(g, rg)][s][0]["_open|" + n] for g in ("conservative", "balanced", "aggressive") for s in ALL[(g, rg)]]))
(HERE / "TRUST_RATES.json").write_text(json.dumps(out, indent=1))
for n in ["MemGate", "S|default", "S|alpha=0.01", "S|alpha=0.02", "S|alpha=0.1", "S|alpha=0.2", "S|minscores=5", "S|minscores=20", "MemGate | no gate", "MemGate | F-test gate"]:
    print(f"{n:24s} PB {100*out['PortBench'].get(n, np.nan):5.1f}%  IB {100*out['InvestorBench'].get(n, np.nan) if n in out['InvestorBench'] else float('nan'):5.1f}%  CTcore {100*out['Controlled core'].get(n, np.nan):5.1f}%  CTinf {100*out['Controlled informative'].get(n, np.nan):5.1f}%")
