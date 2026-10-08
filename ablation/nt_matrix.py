"""Figure 4 matrix (offline, no API calls): every ablation variant on every column, paired against MemTrial.
Real benchmarks: per-seed utility = mean over test dates and investors; two-sided paired t-test over test dates.
PlantedMem: per-seed utility = mean over investors and the regimes of a group; paired t-test over the 100 test seeds.
delta_pct = 100 (variant - MemTrial) / MemTrial; delta_sd = sd over seeds of 100 (variant_s - MemTrial_s) / mean(MemTrial).
'identical' = every (date, investor, seed) value equal to MemTrial's. Writes NT_MATRIX.json."""
import sys, json, glob, re, inspect, pickle, collections, math
from pathlib import Path
import numpy as np
HERE = Path(__file__).resolve().parent; LAB = HERE.parent
sys.path.insert(0, str(LAB / "memtrial")); sys.path.insert(0, str(LAB / "classalloc_and_robustness"))
import memtrial as MT
from paper_tables import pb_load
ROWS = ["MemTrial", "per-experience learner only", "content-aware learner only", "in-sample F-test", "none (always act on values)",
        "never passes (always anchored action)", "fall back to the draft average", "fixed fallback to the reference", "uniform prior"]
PBK = dict(zip(ROWS, ["MemTrial", "MemTrial | identity learner", "MemTrial | content learner only", "MemTrial | F-test gate", "MemTrial | no gate",
                      "MemTrial | gate 0.01", "MemTrial | closed->ensemble", "MemTrial | closed->reference", "MemTrial | uniform prior"]))
IBF = dict(zip(ROWS, ["MemTrial", "MemTrial_identity_learner", "MemTrial_content_learner_only", "MemTrial_F_test_gate", "MemTrial_no_gate",
                      None, "MemTrial_closed_ensemble", "MemTrial_closed_reference", "MemTrial_uniform_prior"]))
CAK = dict(zip(ROWS, ["MemTrial", "identity learner only", "content learner only", "F-test gate", "no gate", "never trust",
                      "closed -> draft average", "closed -> reference", "uniform prior"]))
PMK = dict(zip(ROWS, ["MemTrial", "MemTrial | identity learner", "MemTrial | content learner only", "MemTrial | F-test gate", "MemTrial | no gate",
                      "NT|never trust", "MemTrial | closed->ensemble", "MemTrial | closed->reference", "MemTrial | uniform prior"]))

def p2(d):
    d = np.asarray(d, float); n = len(d); sd = d.std(ddof=1)
    if sd == 0: return 1.0 if abs(d.mean()) == 0 else 0.0
    return float(2 * MT.t_sf(abs(d.mean()) / (sd / math.sqrt(n)), n - 1))

def stats(arr, ref, scale, paired_axis_dates=True):
    """arr, ref: (dates, investors, seeds) for real benchmarks."""
    ps = scale * np.nanmean(arr, (0, 1)); pr = scale * np.nanmean(ref, (0, 1))
    dd = scale * (np.nanmean(arr, (1, 2)) - np.nanmean(ref, (1, 2)))
    return dict(mean=float(ps.mean()), sd=float(ps.std(ddof=1)), delta_pct=float(100 * (ps.mean() - pr.mean()) / pr.mean()),
                delta_sd=float((100 * (ps - pr) / pr.mean()).std(ddof=1)), p=p2(dd),
                identical=bool(np.nanmax(np.abs(arr - ref)) < 1e-12), per_seed=ps.tolist())

OUT = collections.defaultdict(dict)
# ---- PortBench (test months; seeds = deployment worlds)
for cfg, col in (("full-price", "PortBench-Full"), ("raw-price", "PortBench-Raw")):
    A, dates = pb_load(cfg)
    for r in ROWS: OUT[r][col] = stats(np.asarray(A[PBK[r]], float), np.asarray(A[PBK["MemTrial"]], float), 100)
# ---- InvestorBench (fee on the amount traded, as in the paper)
TO = LAB / "investorbench/turnover"
def ibres(r):
    f = HERE / "mt_NT_never_trust.json" if IBF[r] is None else TO / f"mt_{IBF[r]}.json"
    return np.array(json.load(open(f))["res"], float)
for r in ROWS: OUT[r]["InvestorBench"] = stats(ibres(r), ibres("MemTrial"), 1e4)
# ---- ClassAlloc: cb_eval_extra.run_variant (frozen memtrial.py), copied with one change: it also stores its result array
import cb_eval_extra as CX
CX.VARIANTS["never trust"] = dict(learner="auto", alpha=0.0)
src = inspect.getsource(CX.run_variant).replace("    return dict(", "    _STORE[name] = (res.copy(), opened.copy())\n    return dict(", 1)
assert "_STORE[name]" in src
CX._STORE = {}; exec(compile(src, "run_variant_store", "exec"), CX.__dict__)
C = pickle.load(open(LAB / "classalloc_and_robustness/classalloc/run/eval/cache.pkl", "rb"))
for r in ROWS: CX.run_variant(C, CAK[r])
pub = pickle.load(open(LAB / "classalloc_and_robustness/classalloc/run/eval/mt0.pkl", "rb"))[0]
ca_identity = float(np.nanmax(np.abs(CX._STORE["MemTrial"][0] - pub)))
for r in ROWS: OUT[r]["ClassAlloc"] = stats(CX._STORE[CAK[r]][0], CX._STORE["MemTrial"][0], 100)
# ---- PlantedMem (published ct files + the never-trust episodes of nt_pm.py)
CT = collections.defaultdict(dict)
for f in glob.glob(str(LAB / "memtrial/sens/ct_*_*_*.json")):
    g = Path(f).stem.split("_")[1]
    for n, S in json.load(open(f)).items():
        for s, v in S.items(): CT[(g, n)][s] = dict(v[0])
for f in glob.glob(str(HERE / "nt_pm_*_*.json")):
    for g, R in json.load(open(f)).items():
        for n, S in R.items():
            for s, v in S.items(): CT[(g, n)][s].update({k: x for k, x in v[0].items() if k.startswith("NT|")})
G3 = ("conservative", "balanced", "aggressive"); SEEDS = [str(s) for s in range(15, 115)]
GROUPS = {"PlantedMem no signal": ["no influence", "noise only"], "PlantedMem signal": ["beta 0.1", "beta 0.25", "beta 0.5"],
          "PlantedMem informative": ["many experiences, informative content"],
          "PlantedMem core": ["no influence", "noise only", "beta 0.1", "beta 0.25", "beta 0.5"]}
for col, regs in GROUPS.items():
    for r in ROWS:
        x = np.array([[CT[(g, n)][s][PMK[r]] for g in G3 for n in regs] for s in SEEDS]) * 100
        y = np.array([[CT[(g, n)][s][PMK["MemTrial"]] for g in G3 for n in regs] for s in SEEDS]) * 100
        ps, pr = x.mean(1), y.mean(1)
        OUT[r][col] = dict(mean=float(ps.mean()), sd=float(ps.std(ddof=1)), delta_pct=float(100 * (ps.mean() - pr.mean()) / pr.mean()),
                           delta_sd=float((100 * (ps - pr) / pr.mean()).std(ddof=1)), p=p2(ps - pr),
                           identical=bool(np.max(np.abs(x - y)) < 1e-12), per_seed=ps.tolist())
json.dump({"rows": OUT, "ca_identity_vs_published": ca_identity}, open(HERE / "NT_MATRIX.json", "w"))
COLS = ["PortBench-Full", "PortBench-Raw", "InvestorBench", "ClassAlloc", "PlantedMem no signal", "PlantedMem signal", "PlantedMem informative", "PlantedMem core"]
print("CA identity vs published MemTrial:", ca_identity)
for r in ROWS:
    cells = []
    for c in COLS:
        o = OUT[r][c]
        cells.append(f"{o['mean']:.3f}±{o['sd']:.3f}" if r == "MemTrial" else ("=" if o["identical"] else f"{o['delta_pct']:+.1f}±{o['delta_sd']:.1f}{'*' if o['p'] < 0.05 else ''}"))
    print(f"{r:40s} " + " | ".join(cells))
