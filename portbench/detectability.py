"""Does the memory measurably change PortBench's drafts on a date? (Table 1, last column; offline, no LLM calls)
For each configuration, investor and date with drafts for all 16 subsets of the four retrieved experiences: one-way
ANOVA of the drafts' utilities across the 16 subsets (replicate draws within a subset give the noise),
F = variance of the subset means / (pooled within-subset variance x mean(1 / draws per subset)), df = (15, within).
A date counts as detectable when p < 0.05.  usage: python3 detectability.py -> DETECTABILITY.json"""
import sys, json
from pathlib import Path
import numpy as np
HERE = Path(__file__).resolve().parent; sys.path.insert(0, str(HERE)); sys.path.insert(0, str(HERE.parent / "memtrial"))
import monthly_lib as ML
from memtrial import f_sf
MASKS = [str(m) for m in range(16)]

if __name__ == "__main__":
    rows, ids, dates, split = ML.load_monthly(); D = ML.Data(rows, ids, dates, split, cache=ML.load_cache())
    out = {}
    for cfg in D.configs:
        for p in ML.INV:
            S = D.score[(cfg, p)]; kind = f"U_{p}"; P = []
            for d in [d for d in D.dates if d in S and all(any(m in S[d][r] for r in S[d]) for m in MASKS)]:
                T = {m: [S[d][r][m][kind] for r in S[d] if m in S[d][r]] for m in MASKS}
                v = [(np.var(T[m], ddof=1), len(T[m]) - 1) for m in MASKS if len(T[m]) >= 2]
                if not v: continue
                noise, df = sum(a * b for a, b in v) / sum(b for _, b in v), sum(b for _, b in v)
                if not df or not np.isfinite(noise) or noise <= 0: continue
                F = np.var([np.mean(T[m]) for m in MASKS], ddof=1) / (noise * float(np.mean([1.0 / len(T[m]) for m in MASKS])))
                P.append(f_sf(F, 15, df))
            out[f"{cfg}|{p}"] = {"detectable_dates": int(sum(x < 0.05 for x in P)), "dates": len(P)}
            print(cfg, p, out[f"{cfg}|{p}"])
    (HERE / "DETECTABILITY.json").write_text(json.dumps(out, indent=1))
