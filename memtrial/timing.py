"""Learning overhead of MemGate per decision (python3 timing.py): wall time of MemGateBank.decide + matured with the
single frozen variant, inside PlantedMem episodes (M = 12 and M = 48 experiences). Single process."""
import sys, time, json
from pathlib import Path
import numpy as np
HERE = Path(__file__).resolve().parent; sys.path.insert(0, str(HERE)); sys.path.insert(0, str(HERE.parent / "plantedmem"))
import memgate as MGm
import suite as V5
T = {"decide": [], "matured": [], "H": []}
class Timed(MGm.MemGateBank):
    def decide(self, *a):
        t0 = time.perf_counter(); o = super().decide(*a); T["decide"].append(time.perf_counter() - t0); T["H"].append(len({h for _, h, _ in self.records})); return o
    def matured(self, *a):
        t0 = time.perf_counter(); super().matured(*a); T["matured"].append(time.perf_counter() - t0)
V5.MemGateBank = lambda Z, V, masks=None: Timed(Z, {"MemGate": dict(learner="auto")}, masks=masks)
out = {}
for n in ("beta 0.25", "many experiences, informative content"):
    for k in T: T[k].clear()
    for s in range(15, 20): V5.episode(seed=s, gamma=5.0, **V5.REGIMES[n])
    d = np.array(T["decide"]) * 1000; m = np.array(T["matured"]) * 1000
    out[n] = dict(decide_ms_mean=float(d.mean()), decide_ms_p95=float(np.percentile(d, 95)), matured_ms_mean=float(m.mean()), max_experiences_seen=int(max(T["H"])), decisions=len(d))
print(json.dumps(out, indent=1)); (HERE / "TIMING.json").write_text(json.dumps(out, indent=1))
