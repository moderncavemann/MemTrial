#!/usr/bin/env python3
"""Collect InvestorBench and ClassAlloc results for every LLM and temperature (offline) -> RESULTS_MODELS.json.
Reads the evaluation outputs of ib_turnover.py / ib_eval_variant.py (InvestorBench) and cb_eval.py (ClassAlloc), and the
three-draw Self-consistency of sc_eval.py when its extra draws are complete. Per seed: mean over test dates and investors;
reported: mean and sd over the three seeds; p: one-sided paired t-test over dates (as in the paper)."""
import json, math, pickle, sys
from pathlib import Path
import numpy as np
HERE = Path(__file__).resolve().parent; LAB = HERE.parent
sys.path.insert(0, str(LAB / "memtrial")); import memgate as MG
EXP = ["FinMem", "MemRL", "Reflexion", "ExpeL"]
LLM_BASE = ["Zero-shot (no memory)", "Self-consistency", "Similarity retrieval (top-2)", "Similarity retrieval (top-4)", "FinMem", "MemRL",
            "Reflexion", "ExpeL", "Draft averaging", "Counterfactual selection", "Uplift credit", "Hedge"]
IB_RUNS = {"main": ("gpt-4.1-mini", 0.7), "t0.0": ("gpt-4.1-mini", 0.0), "t0.3": ("gpt-4.1-mini", 0.3), "t1.0": ("gpt-4.1-mini", 1.0),
           "nano": ("gpt-4.1-nano", 0.7), "gpt5mini": ("gpt-5-mini", None), "llama70b": ("Llama-3.3-70B-Instruct", 0.7),
           "gemini25flash": ("Gemini 2.5 Flash", 0.7), "qwen3": ("Qwen3-235B-A22B-Instruct-2507", 0.7),
           "deepseekv31": ("DeepSeek-V3.1", 0.7), "haiku45": ("Claude Haiku 4.5", 0.7)}
CB_RUNS = {"main": ("gpt-4.1-mini", "2024-06"), "gpt-4.1-nano": ("gpt-4.1-nano", "2024-06"), "gpt-5-mini": ("gpt-5-mini", "2024-05"),
           "llama-3.3-70b": ("Llama-3.3-70B-Instruct", "2023-12"), "gemini-2.5-flash": ("Gemini 2.5 Flash", "2025-01"),
           "qwen3-235b": ("Qwen3-235B-A22B-Instruct-2507", None), "deepseek-v3.1": ("DeepSeek-V3.1", None),
           "claude-haiku-4.5": ("Claude Haiku 4.5", "2025-07"),     # published cutoffs of the training data (None: not published);
           "masked-gpt-4.1-mini": ("gpt-4.1-mini, dates hidden", "2024-06"), "masked-gpt-5-mini": ("gpt-5-mini, dates hidden", "2024-05")}  # run_cb_masked.py
# Claude Haiku 4.5: trained on data up to July 2025 (its "reliable knowledge cutoff" is February 2025)


def paired_p(a, b):
    d = np.asarray(a) - np.asarray(b); n = len(d); sd = d.std(ddof=1)
    if sd == 0: return 0.0 if d.mean() > 0 else 1.0
    return float(MG.t_sf(d.mean() / (sd / math.sqrt(n)), n - 1))


def ms(x): x = np.asarray(x, float); return dict(mean=float(x.mean()), sd=float(x.std(ddof=1)), per_seed=x.tolist())


def summarize(R, opened, scale, sel=None):
    """R: method -> array (dates x investors x seeds); opened: method -> same shape (0/1)."""
    sel = slice(None) if sel is None else sel
    per = {k: scale * np.nanmean(A[sel], (0, 1)) for k, A in R.items()}
    day = {k: scale * np.nanmean(A[sel], (1, 2)) for k, A in R.items()}
    rows = {k: ms(v) for k, v in per.items()}
    best = max(EXP, key=lambda k: rows[k]["mean"]); other = max(LLM_BASE, key=lambda k: rows[k]["mean"])
    mg = "MemGate"
    out = dict(rows=rows, best_experience_agent=best, best_other_llm=other,
               improv=ms(per[mg] - per[best]), p_vs_best=paired_p(day[mg], day[best]),
               mg_minus_1N=ms(per[mg] - per["1/N"]), best_minus_1N=float(rows[best]["mean"] - rows["1/N"]["mean"]),
               n_dates=int(np.asarray(R["1/N"][sel]).shape[0]))
    if opened:
        out["trust_rate_pct"] = {k: ms(100 * np.mean(np.asarray(v)[sel], (0, 1))) for k, v in opened.items()}
    return out


def sc_new(bench, run):
    f = HERE / "sc_runs" / f"{bench}_{run}" / "SC_EVAL.json"
    if not f.exists(): return None
    J = json.load(open(f))
    if J.get("mock") or J.get("draws_per_seed", 0) < 2.9: return {"incomplete": True, "draws_per_seed": J.get("draws_per_seed")}
    return J


res = {"investorbench": {}, "classalloc": {}}
for run, (model, T) in IB_RUNS.items():
    ev = (LAB / "investorbench" / "turnover") if run == "main" else (HERE / "ib_runs" / run / "eval")
    if not (ev / "mg_MemGate.json").exists() or not (ev / "base.json").exists(): res["investorbench"][run] = None; continue
    B = json.load(open(ev / "base.json")); R = {k: np.array(v, float) for k, v in B["res"].items()}; op = {}
    for name in ("MemGate", "MemGate | no gate"):
        J = json.load(open(ev / f"mg_{name.replace(' | ', '_').replace(' ', '_')}.json")); R[name] = np.array(J["res"], float); op[name] = np.array(J["open"], float)
    sc = sc_new("ib", run)
    if sc and not sc.get("incomplete"): R["Self-consistency"] = np.array(sc["res_new"], float) / 1e4
    out = summarize(R, op, 1e4); out.update(model=model, temperature=T, unit="bp per day", sc_three_draws=bool(sc and not sc.get("incomplete")))
    res["investorbench"][run] = out
for run, (model, cut) in CB_RUNS.items():
    folder = HERE / "classalloc" / ("run" if run == "main" else f"run_{run}_t0.7")
    ev = folder / "eval"
    if not (ev / "mg0.pkl").exists(): res["classalloc"][run] = None; continue
    base = pickle.load(open(ev / "base.pkl", "rb")); R = dict(base[0]); op = {}
    for i, name in enumerate(("MemGate", "MemGate | no gate", "MemGate | F-test gate")):
        r, c, o, fs = pickle.load(open(ev / f"mg{i}.pkl", "rb")); R[name] = r; op[name] = o
    dates = pickle.load(open(ev / "cache.pkl", "rb"))["dates"]
    sc = sc_new("cb", run)
    if sc and not sc.get("incomplete"): R["Self-consistency"] = np.array(sc["res_new"], float) / 100
    out = summarize(R, op, 100); out.update(model=model, cutoff=cut, unit="pp per month", sc_three_draws=bool(sc and not sc.get("incomplete")))
    for lab, c in (("gpt-4.1-mini cutoff", "2024-06"), ("own cutoff", cut)):
        if c is None: continue
        post = np.array([t[:7] > c for t in dates])
        out[f"before {lab}"] = summarize(R, op, 100, ~post); out[f"after {lab}"] = summarize(R, op, 100, post)
    res["classalloc"][run] = out
(HERE / "RESULTS_MODELS.json").write_text(json.dumps(res, indent=1))
for b, unit in (("investorbench", "bp/day"), ("classalloc", "pp/month")):
    print(f"\n{b} ({unit}): run | 1/N | zero-shot | SC (3 draws?) | best exp agent | MemTrial | improv (p) | trust")
    for run, o in res[b].items():
        if o is None: print(f"  {run:16s} (not evaluated yet)"); continue
        r = o["rows"]; f = lambda k: f"{r[k]['mean']:.3f}±{r[k]['sd']:.3f}"
        print(f"  {run:16s} {r['1/N']['mean']:.3f} | {f('Zero-shot (no memory)')} | {f('Self-consistency')} ({'y' if o['sc_three_draws'] else 'n'}) | "
              f"{o['best_experience_agent']} {f(o['best_experience_agent'])} | {f('MemGate')} | {o['improv']['mean']:+.3f}±{o['improv']['sd']:.3f} ({o['p_vs_best']:.3f}) | "
              f"{o['trust_rate_pct']['MemGate']['mean']:.1f}±{o['trust_rate_pct']['MemGate']['sd']:.1f}")
