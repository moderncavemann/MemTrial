"""Never-trust ablation in PlantedMem (offline, no API calls).

"NT|never trust" is MemTrial (MemGate) with the trust test never passing (alpha = 0), so every decision takes the anchored
action; "NT|never trust, 1/N reference" is the same with 1/N as the reference (as W1N|MemGate). Same episodes, seeds and
code path as memtrial/sens.py run_ct (SensBank over the frozen memgate.py, which is not modified); only variants are added,
and the bank's learning does not depend on which variant is deployed, so all other values are unchanged.
usage: python3 nt_pm.py <s0> <s1> [procs]   -> nt_pm_<s0>_<s1>.json (three investors, seven regimes)"""
import sys, json, time, collections
from pathlib import Path
HERE = Path(__file__).resolve().parent; LAB = HERE.parent
for p in (LAB / "plantedmem", LAB / "plantedmem", LAB / "memtrial"): sys.path.insert(0, str(p))
import sens as SN
import suite as V5
NT_V = {"NT|never trust": dict(learner="auto", alpha=0.0),
        "NT|never trust, 1/N reference": dict(learner="auto", closed="ftrl1n", alpha=0.0)}
V5.MemGateBank = lambda Z, V, masks=None: SN.SensBank(Z, {**V5.MG_VARIANTS, **SN.W1N_V, **NT_V}, masks=masks, extra="ct1n")
KEEP = ("1/N", "no-memory 8-draw ensemble (reference)", "uniform aggregation of the 8 members", "counterfactual credit argmax top-2")

def job(a):
    g, n, s = a
    o = V5._job(a)
    return a, [{k: v for k, v in o[0].items() if "|" in k or k.startswith("MemGate") or k in KEEP}, o[1]]

if __name__ == "__main__":
    s0, s1 = int(sys.argv[1]), int(sys.argv[2]); procs = int(sys.argv[3]) if len(sys.argv) > 3 else 4
    import multiprocessing as mp
    jobs = [(g, n, s) for g in V5.GAMMA for n in V5.REGIMES for s in range(s0, s1)]; t0 = time.time()
    R = collections.defaultdict(lambda: collections.defaultdict(dict)); done = 0
    with mp.get_context("fork").Pool(procs) as pool:
        for (g, n, s), o in pool.imap_unordered(job, jobs, chunksize=4):
            R[g][n][str(s)] = o; done += 1
            if done % 100 == 0: print(f"{done}/{len(jobs)} episodes, {time.time() - t0:.0f}s", flush=True)
    json.dump(R, open(HERE / f"nt_pm_{s0}_{s1}.json", "w")); print("done", f"{time.time() - t0:.0f}s", flush=True)
