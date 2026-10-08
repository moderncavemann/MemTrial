"""Adds the row 'no online learning (q fixed at alpha0 = 0.9)' to NT_MATRIX.json -> NT_MATRIX2.json (same statistics)."""
import sys, json, glob, collections, math
from pathlib import Path
import numpy as np
HERE = Path(__file__).resolve().parent; LAB = HERE.parent
sys.path.insert(0, str(LAB / "memtrial")); import memtrial as MT
from paper_tables import pb_load
M = json.load(open(HERE / "NT_MATRIX.json"))
def p2(d):
    d = np.asarray(d, float); n = len(d); sd = d.std(ddof=1)
    if sd == 0: return 1.0 if abs(d.mean()) == 0 else 0.0
    return float(2 * MT.t_sf(abs(d.mean()) / (sd / math.sqrt(n)), n - 1))
def stats(arr, ref, scale):
    ps = scale * np.nanmean(arr, (0, 1)); pr = scale * np.nanmean(ref, (0, 1)); dd = scale * (np.nanmean(arr, (1, 2)) - np.nanmean(ref, (1, 2)))
    return dict(mean=float(ps.mean()), sd=float(ps.std(ddof=1)), delta_pct=float(100 * (ps.mean() - pr.mean()) / pr.mean()),
                delta_sd=float((100 * (ps - pr) / pr.mean()).std(ddof=1)), p=p2(dd), identical=bool(np.nanmax(np.abs(arr - ref)) < 1e-12), per_seed=ps.tolist())
row = {}
for cfg, col in (("full-price", "PortBench-Full"), ("raw-price", "PortBench-Raw")):
    A, dates = pb_load(cfg); G = ("conservative", "balanced", "aggressive")
    N = {p: json.load(open(HERE / f"nt_pb_{cfg}_{p}.json")) for p in G}
    X = np.array([[N[p]["res"]["NT|fixed q=0.9"][d] for p in G] for d in dates], float)
    row[col] = stats(X, np.asarray(A["MemTrial"], float), 100)
IB = json.load(open(HERE / "mt_NT_fixed_q_0_9.json")); MTI = json.load(open(LAB / "investorbench/turnover/mt_MemTrial.json"))
row["InvestorBench"] = stats(np.array(IB["res"], float), np.array(MTI["res"], float), 1e4)
CA = json.load(open(HERE / "nt_fixedq_ib_cb.json"))["ca"]
row["ClassAlloc"] = stats(np.array(CA["NT|fixed q=0.9"], float), np.array(CA["MemTrial"], float), 100)
CT = collections.defaultdict(dict)
for f in glob.glob(str(LAB / "memtrial/sens/ct_*_*_*.json")):
    g = Path(f).stem.split("_")[1]
    for n, S in json.load(open(f)).items():
        for s, v in S.items(): CT[(g, n)][s] = dict(v[0])
for f in glob.glob(str(HERE / "nt_pmq_*_*.json")):
    for g, R in json.load(open(f)).items():
        for n, S in R.items():
            for s, v in S.items(): CT[(g, n)][s].update({k: x for k, x in v[0].items() if k.startswith("NT|")})
G3 = ("conservative", "balanced", "aggressive"); SEEDS = [str(s) for s in range(15, 115)]
GROUPS = {"PlantedMem no signal": ["no influence", "noise only"], "PlantedMem signal": ["beta 0.1", "beta 0.25", "beta 0.5"],
          "PlantedMem informative": ["many experiences, informative content"], "PlantedMem core": ["no influence", "noise only", "beta 0.1", "beta 0.25", "beta 0.5"]}
for col, regs in GROUPS.items():
    x = np.array([[CT[(g, n)][s]["NT|fixed q=0.9"] for g in G3 for n in regs] for s in SEEDS]) * 100
    y = np.array([[CT[(g, n)][s]["MemTrial"] for g in G3 for n in regs] for s in SEEDS]) * 100
    a, b = x.mean(1), y.mean(1)
    row[col] = dict(mean=float(a.mean()), sd=float(a.std(ddof=1)), delta_pct=float(100 * (a.mean() - b.mean()) / b.mean()),
                    delta_sd=float((100 * (a - b) / b.mean()).std(ddof=1)), p=p2(a - b), identical=bool(np.max(np.abs(x - y)) < 1e-12), per_seed=a.tolist())
# weak reference (1/N as the reference), core regimes: MemTrial, fixed q, always 1/N, never trust
W = {}
for k in ("W1N|MemTrial", "NT|fixed q=0.9, 1/N reference", "W1N|fallback to 1/N", "W1N|uniform prior"):
    W[k] = (100 * np.array([np.mean([CT[(g, n)][s][k] for g in G3 for n in GROUPS["PlantedMem core"]]) for s in SEEDS])).tolist()
M["rows"]["no online learning"] = row; M["weakref_core_per_seed"] = W
json.dump(M, open(HERE / "NT_MATRIX2.json", "w"))
for c, o in row.items(): print(f"{c:24s} {o['mean']:.4f} ± {o['sd']:.4f}  Δ {o['delta_pct']:+.2f} ± {o['delta_sd']:.2f}  p {o['p']:.3g}  identical {o['identical']}")
for k, v in W.items(): print(k, round(float(np.mean(v)), 4))
