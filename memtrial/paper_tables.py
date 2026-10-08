"""Paper tables (SPRINT-style): every method on every benchmark, mean +- sd over seeds (usage: python3 paper_tables.py) -> TABLES.md.
PortBench-Full / PortBench-Raw: 3 seeds (independent draws per arm), test months 2023-2024, averaged over the 3 investors.
InvestorBench: 3 seeds, test days 2020-10..2021-05, averaged over the 3 investors (if the run exists).
Controlled (= PlantedMem): 100 test seeds, averaged over the 5 core regimes and the 3 investors.
* = MemGate better than the best experience-learning agent of the column, paired t-test p < 0.05 (dates for real
benchmarks, seeds for the controlled one)."""
import sys, json, glob, math, collections
from pathlib import Path
import numpy as np
HERE = Path(__file__).resolve().parent; LAB = HERE.parent
sys.path.insert(0, str(HERE))
from memgate import t_sf

FAMILIES = [("Rule-based portfolios", ["1/N", "Minimum variance"]),
            ("LLM agents without experience learning", ["Zero-shot (no memory)", "Self-consistency", "Similarity retrieval (top-2)", "Similarity retrieval (top-4)"]),
            ("Experience-learning agents", ["FinMem", "MemRL", "Reflexion", "ExpeL"]),
            ("Credit-based and ensemble learners", ["Uplift credit", "Counterfactual selection", "Draft averaging", "Hedge"]),
            ("Ours", ["MemGate"])]
EXPERIENCE = ["FinMem", "MemRL", "Reflexion", "ExpeL"]
PB = {"1/N": "1/N", "Minimum variance": "Minimum variance", "Zero-shot (no memory)": "Zero-shot (no memory)", "Self-consistency": "Self-consistency (3 drafts)",
      "Similarity retrieval (top-2)": "Similarity retrieval (top-2)", "Similarity retrieval (top-4)": "Similarity retrieval (top-4)", "FinMem": "FinMem",
      "MemRL": "MemRL", "Reflexion": "Reflexion", "ExpeL": "ExpeL (adapted)", "Uplift credit": "Uplift credit (UpliftMem-style)",
      "Counterfactual selection": "Counterfactual selection (no gate)", "Draft averaging": "Draft averaging (8 drafts)", "Hedge": "Hedge (memory families)", "MemGate": "MemGate"}
CT = {"1/N": "1/N", "Minimum variance": "Minimum variance", "Zero-shot (no memory)": "no memory", "Self-consistency": "no-memory 8-draw ensemble (reference)",
      "Similarity retrieval (top-2)": "similarity top-2", "Similarity retrieval (top-4)": "all 4", "FinMem": "FinMem (adapted)",
      "MemRL": "outcome credit top-2 (MemRL/FinMem-style)", "Reflexion": "Reflexion (adapted)", "ExpeL": "ExpeL (adapted)",
      "Uplift credit": "uplift credit top-2 (regression-adjusted, UpliftMem/UCOB-style)", "Counterfactual selection": "counterfactual credit argmax top-2",
      "Draft averaging": "uniform aggregation of the 8 members", "Hedge": "Hedge (adapted)", "MemGate": "MemGate"}
ADAPTED = {"PortBench": {"ExpeL", "Uplift credit"}, "Controlled": {"FinMem", "MemRL", "Reflexion", "ExpeL", "Hedge", "Uplift credit"}, "InvestorBench": {"FinMem", "MemRL", "Uplift credit"}}
SCALE = {"InvestorBench": 1e4}          # InvestorBench in basis points per day; the others in pp per period
CORE = ["no influence", "noise only", "beta 0.1", "beta 0.25", "beta 0.5"]


def pb_load(cfg):
    """{method: array (dates, investors, seeds)} on test dates."""
    R = {p: json.load(open(LAB / "portbench" / f"pb_all_{cfg}_{p}.json")) for p in ("conservative", "balanced", "aggressive")}
    split = R["balanced"]["split"]; dates = sorted(d for d in R["balanced"]["res"]["1/N"] if d >= split)
    out = {}
    for k in set.intersection(*[set(R[p]["res"]) for p in R]):
        out[k] = np.array([[R[p]["res"][k][d] for p in R] for d in dates])
    return out, dates


def ct_load(pattern="suite5_"):
    ALL = collections.defaultdict(dict)
    for f in glob.glob(str(LAB / "plantedmem" / "results" / f"{pattern}*_*_*.json")):
        g = Path(f).stem.split("_")[1]
        for n, S in json.load(open(f)).items(): ALL[(g, n)].update({s: v for s, v in S.items() if int(s) >= 15})
    return ALL


def ct_matrix(ALL, key, regs=CORE):
    gs = ("conservative", "balanced", "aggressive"); seeds = sorted(ALL[("balanced", regs[0])], key=int)
    return np.array([[[ALL[(g, n)][s][0][key] for g in gs] for n in regs] for s in seeds])     # (seeds, regimes, investors)


def ib_load():
    f = LAB / "investorbench" / "ib_all.json"
    if not f.exists(): return None
    J = json.load(open(f)); return {k: np.array(v, float) for k, v in J["res"].items()}, J["dates"]


def fmt(m, s, bold=False, star=False, sc=100):
    x = f"{sc*m:.2f} ± {sc*s:.2f}" + ("*" if star else "")
    return f"**{x}**" if bold else x


def ptest(a, b):
    d = np.asarray(a) - np.asarray(b); n = len(d)
    if n < 3 or d.std(ddof=1) == 0: return 1.0
    return t_sf(d.mean() / (d.std(ddof=1) / math.sqrt(n)), n - 1)


def main():
    cols = []                                           # (name, {method: (mean, sd, per-unit array for tests)})
    for cfg, name in (("full-price", "PortBench-Full"), ("raw-price", "PortBench-Raw")):
        A, dates = pb_load(cfg); col = {}
        for lab, k in PB.items():
            if k not in A: continue
            x = A[k]; per_seed = x.mean((0, 1)); per_date = x.mean((1, 2))
            col[lab] = (per_seed.mean(), per_seed.std(ddof=1), per_date)
        cols.append((name, col))
    ib = ib_load()
    if ib:
        A, dates = ib; col = {}
        for lab in PB:
            k = lab if lab in A else None
            if k is None: continue
            x = A[k]; per_seed = x.mean((0, 1)); per_date = x.mean((1, 2)); col[lab] = (per_seed.mean(), per_seed.std(ddof=1), per_date)
        cols.append(("InvestorBench", col))
    ALL = ct_load(); col = {}
    for lab, k in CT.items():
        x = ct_matrix(ALL, k); per_seed = x.mean((1, 2)); col[lab] = (per_seed.mean(), per_seed.std(ddof=1), per_seed)
    cols.append(("Controlled", col))
    L = ["# Main results (mean ± sd over seeds; utility in percentage points per decision period)", "",
         "| Method | " + " | ".join(c for c, _ in cols) + " |", "|---" * (len(cols) + 1) + "|"]
    best = {c: max((v[0], k) for k, v in col.items() if k != "MemGate")[1] for c, col in cols}
    for fam, methods in FAMILIES:
        L.append(f"| *{fam}* |" + " |" * len(cols))
        for m in methods:
            cells = []
            for c, col in cols:
                if m not in col: cells.append("—"); continue
                mu, sd, per = col[m]; star = False
                if m == "MemGate":
                    be = max((col[e][0], e) for e in EXPERIENCE if e in col)[1]; star = ptest(per, col[be][2]) < 0.05 and mu > col[be][0]
                bold = mu >= max(v[0] for v in col.values()) - 1e-12
                tag = "†" if m in ADAPTED.get(c.split("-")[0], set()) else ""
                cells.append(fmt(mu, sd, bold, star, SCALE.get(c, 100)) + tag)
            L.append(f"| {m} | " + " | ".join(cells) + " |")
    imp, gap = [], []
    for c, col in cols:
        be = max((col[e][0], e) for e in EXPERIENCE if e in col)[1]; d = col["MemGate"][0] - col[be][0]
        sc = SCALE.get(c, 100)
        imp.append(f"{sc*d:+.2f} (vs {be}; p = {ptest(col['MemGate'][2], col[be][2]):.3f})")
        gap.append(f"{sc*(col['MemGate'][0] - col['1/N'][0]):+.2f}")
    L += [f"| Improv. over best experience-learning agent | " + " | ".join(imp) + " |", f"| MemGate minus 1/N | " + " | ".join(gap) + " |", "",
          "† adapted: the baseline's memory-scoring rule re-implemented over the same experience pool (its own memory module cannot run there)."]
    # ---- ablation ----
    ABL_NAME = {"MemGate": "MemGate (full)", "MemGate | identity learner": "w/o content-based value learning (per-experience only)",
                "MemGate | content learner only": "w/o learner selection (content-based only)", "MemGate | F-test gate": "In-sample test instead of forward validation",
                "MemGate | no gate": "w/o trust gate", "MemGate | closed->ensemble": "Fallback to the draft average instead of 1/N",
                "MemGate | closed->reference": "Fallback to 1/N without online learning", "MemGate | uniform prior": "w/o anchoring (uniform prior)"}
    ABL = ["MemGate", "MemGate | identity learner", "MemGate | content learner only", "MemGate | F-test gate", "MemGate | no gate",
           "MemGate | closed->ensemble", "MemGate | closed->reference", "MemGate | uniform prior"]
    L += ["", "# Ablation (mean ± sd over seeds; same units)", "", "| Variant | PortBench-Full | PortBench-Raw | InvestorBench (bp/day) | Controlled (core) | Controlled (many experiences, informative content) |", "|---|---|---|---|---|---|"]
    IBA = ib_load()
    PBA = {cfg: pb_load(cfg)[0] for cfg in ("full-price", "raw-price")}
    for v in ABL:
        cells = []
        for cfg in ("full-price", "raw-price"):
            x = PBA[cfg][v].mean((0, 1)); cells.append(fmt(x.mean(), x.std(ddof=1)))
        if IBA:
            x = np.nanmean(IBA[0][v], (0, 1)); cells.append(fmt(x.mean(), x.std(ddof=1), sc=1e4))
        x = ct_matrix(ALL, v).mean((1, 2)); cells.append(fmt(x.mean(), x.std(ddof=1)))
        x = ct_matrix(ALL, v, ["many experiences, informative content"]).mean((1, 2)); cells.append(fmt(x.mean(), x.std(ddof=1)))
        L.append(f"| {ABL_NAME[v]} | " + " | ".join(cells) + " |")
    (HERE / "TABLES.md").write_text("\n".join(L)); print("\n".join(L))


if __name__ == "__main__":
    main()
