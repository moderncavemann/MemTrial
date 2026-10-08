"""Daily net returns and cumulative wealth on InvestorBench (offline, no API calls; usage: python3 wealth_ib.py).

Reads the published evaluation files (never writes to them):
  main run  : ../investorbench/turnover/{base.json, mg_MemGate.json}       (gpt-4.1-mini, T = 0.7)
  variants  : ../classalloc_and_robustness/ib_runs/<v>/eval/{base.json, mg_MemGate.json}  (other LLMs and temperatures)
Net return of a day = gross return - fee on the traded amount (15 bp), exactly the `net` inside ib_turnover.util.
Baselines store gross and cost; for MemTrial (file name mg_MemGate.json) the net return is recovered from the stored
utility u = net - gamma/2 * net^2 (the root near zero); the same inversion is checked on every baseline against gross - cost.
Writes WEALTH_IB.json."""
import json
from pathlib import Path
import numpy as np
HERE = Path(__file__).resolve().parent; ML = HERE.parent
GAM = [10.0, 5.0, 2.0]; INVN = ["conservative", "balanced", "aggressive"]
RUNS = {"gpt-4.1-mini (T=0.7)": ML / "investorbench/turnover"}
for v, lab in [("nano", "gpt-4.1-nano (T=0.7)"), ("gpt5mini", "gpt-5-mini"), ("llama70b", "Llama-3.3-70B (T=0.7)"),
               ("t0.0", "gpt-4.1-mini (T=0.0)"), ("t0.3", "gpt-4.1-mini (T=0.3)"), ("t1.0", "gpt-4.1-mini (T=1.0)"),
               ("gemini25flash", "Gemini 2.5 Flash (T=0.7)"), ("qwen3", "Qwen3-235B (T=0.7)"), ("deepseekv31", "DeepSeek-V3.1 (T=0.7)"),
               ("haiku45", "Claude Haiku 4.5 (T=0.7)")]:
    if (ML / f"classalloc_and_robustness/ib_runs/{v}/eval/mg_MemGate.json").exists():    # runs evaluated so far
        RUNS[lab] = ML / f"classalloc_and_robustness/ib_runs/{v}/eval"
KEEP = ["1/N", "Minimum variance", "Zero-shot (no memory)", "Self-consistency", "FinMem", "MemRL", "Reflexion", "ExpeL",
        "Uplift credit", "Counterfactual selection", "Draft averaging", "Hedge"]


def inv_net(u, gam): return (1.0 - np.sqrt(1.0 - 2.0 * gam * u)) / gam


def stats(net):
    """net: (days, seeds) daily net returns -> per-seed statistics, then mean and sd over seeds."""
    W = np.cumprod(1 + net, 0)
    peak = np.maximum.accumulate(np.vstack([np.ones((1, net.shape[1])), W]), 0)[1:]
    mdd = (1 - W / peak).max(0)
    ann = W[-1] ** (252 / len(net)) - 1
    vol = net.std(0, ddof=1) * np.sqrt(252); shp = net.mean(0) / net.std(0, ddof=1) * np.sqrt(252)
    f = lambda a: [float(np.mean(a)), float(np.std(a, ddof=1))]
    return {"terminal": f(W[-1]), "ann_return": f(ann), "ann_vol": f(vol), "sharpe": f(shp), "max_drawdown": f(mdd)}


def main():
    dates = json.load(open(ML / "investorbench/ib_all_turnover.json"))["dates"]
    out = {"dates": dates, "investors": INVN, "runs": {}, "check": {}}
    for lab, d in RUNS.items():
        B = json.load(open(d / "base.json")); Mg = json.load(open(d / "mg_MemGate.json"))
        net = {}; worst = 0.0
        for k in KEEP:
            g = np.array(B["gross"][k], float); c = np.array(B["cost"][k], float); u = np.array(B["res"][k], float)
            net[k] = g - c
            for pi in range(3): worst = max(worst, float(np.nanmax(np.abs(inv_net(u[:, pi], GAM[pi]) - net[k][:, pi]))))
        u = np.array(Mg["res"], float); net["MemTrial"] = np.stack([inv_net(u[:, pi], GAM[pi]) for pi in range(3)], 1)
        assert not any(np.isnan(v).any() for v in net.values()), lab
        out["check"][lab] = worst
        R = {}
        for k, a in net.items():
            R[k] = {}
            for pi, p in enumerate(INVN):
                W = np.cumprod(1 + a[:, pi, :], 0)
                R[k][p] = {"wealth_mean": W.mean(1).tolist(), "wealth_sd": W.std(1, ddof=1).tolist(),
                           "net_by_seed": a[:, pi, :].tolist(), "stats": stats(a[:, pi, :])}
        out["runs"][lab] = R
        print(f"{lab:24s} inversion check max |err| = {worst:.2e}")
        for p in INVN:
            print("   ", p, "  ".join(f"{k.split(' (')[0][:10]}={R[k][p]['stats']['terminal'][0]:.3f}" for k in
                                   ["1/N", "Zero-shot (no memory)", "FinMem", "MemRL", "Reflexion", "ExpeL", "MemTrial"]))
    (HERE / "WEALTH_IB.json").write_text(json.dumps(out))


if __name__ == "__main__":
    main()
