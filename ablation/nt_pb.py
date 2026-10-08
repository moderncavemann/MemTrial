"""PortBench: MemTrial and the anchored action without online learning (q fixed at the prior alpha0 = 0.9), offline.
Same code path as memtrial/sens.py run_pb (SensBank over the frozen memgate.py, unmodified; a variant with lam0 = 1e9
keeps q at alpha0). usage: python3 nt_pb.py <cfg> <investor>  -> nt_pb_<cfg>_<investor>.json"""
import sys, os, json, time
from pathlib import Path
HERE = Path(__file__).resolve().parent; LAB = HERE.parent
for p in (LAB / "memtrial", LAB / "portbench"): sys.path.insert(0, str(p))
import sens as SN
import pb_all as PA
V = {"MemGate": dict(learner="auto"), "NT|fixed q=0.9": dict(learner="auto", a0=0.9, lam0=1e9)}
PA.MemGateBank = lambda Z, VV, masks=None: SN.SensBank(Z, V, masks=masks)
cfg, p = sys.argv[1], sys.argv[2]; t0 = time.time()
rows, ids, dates, split = PA.ML.load_monthly(); inp = json.load(open(PA.HERE / "data/inputs.json"))
D = PA.ML.Data(rows, ids, dates, split, cache=PA.ML.load_cache())
Z = PA.content_features(D, inp); t1 = time.time()
res, gl = PA.run(D, cfg, p, Z)
keep = {k: v for k, v in res.items() if k in V or k in ("1/N", "Draft averaging (8 drafts)")}
json.dump({"res": keep, "split": split}, open(HERE / f"nt_pb_{cfg}_{p}.json", "w"))
print(cfg, p, f"load {t1 - t0:.0f}s run {time.time() - t1:.0f}s", flush=True)
