"""Adds the k=4 full-factorial design (16 drafts) to the k sensitivity of k_designs.py, on the same worlds and draws
(python3 k_designs16.py prun <investor> <s0> <s1>) -> results/k16_<investor>_<s0>_<s1>.json.
The design uses bases[0..15] of the same 32 pre-drawn bases per date, so it shares common random numbers with k_designs.py;
adding it does not change any other design. k_designs.py and memtrial.py are unmodified."""
import sys, json, time, collections
import multiprocessing as mp
import k_designs as K
K.DESIGNS = {"k=4, full (16 drafts)": (4, list(range(16)))}
if __name__ == "__main__":
    gname, s0, s1 = sys.argv[2], int(sys.argv[3]), int(sys.argv[4]); t0 = time.time()
    jobs = [(gname, n, s) for n in K.S.REGIMES for s in range(s0, s1)]
    with mp.get_context("fork").Pool(4) as pool: outs = pool.map(K._job, jobs, chunksize=2)
    R = collections.defaultdict(dict)
    for (g_, n, s), o in zip(jobs, outs): R[n][str(s)] = o
    json.dump(R, open(K.OUT / f"k16_{gname}_{s0}_{s1}.json", "w")); print(gname, s0, s1, f"{time.time() - t0:.0f}s", flush=True)
