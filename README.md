# MemTrial: Learning When to Trust Memory in LLM Portfolio Agents

Code for the paper *MemTrial: Learning When to Trust Memory in LLM Portfolio Agents*: the method, the four benchmarks with
every baseline and ablation, the LLM runners, and the logged LLM outputs from which every number in the paper's tables is
recomputed offline.
In the code, MemTrial appears under its working name **MemGate** (`memtrial/memgate.py`, class `MemGateBank`; result keys
`MemGate` and `MemGate | <variant>`), and the conservative reference of the paper is called the safe reference.

## Requirements

Python 3.10+ with numpy and pandas (`pip install -r requirements.txt`). No GPU. The LLM runners use only the standard library.

## Reproduce the results (offline, no API key)

```bash
bash reproduce.sh                  # about two hours on 4 cores; every step can also be run on its own
python3 tables/print_tables.py     # Tables 1 and 3 and the ablation of Figure 4 in the paper's layout -> tables/PAPER_TABLES.md
python3 wealth/return_risk.py      # Table 4 (return and risk) -> wealth/RETURN_RISK.json
```

`reproduce.sh` lists every step in order. The first PortBench step (`python3 portbench/monthly_lib.py`, about 30 minutes,
resumable) computes the investor projections of all logged drafts. Running these scripts on the included data gives our
results: Tables 1, 3 and 4 and Figure 4 are reproduced to the last reported digit.

## Layout

| Folder | Contents | Paper |
|---|---|---|
| `memtrial/` | `memgate.py`: MemTrial (Banzhaf contributions over the half-fraction design, Fay-Herriot value models, prequential trust test, KL-anchored action), frozen before InvestorBench and ClassAlloc were run (hash in `MEMGATE_FROZEN.json`). `sens.py`: hyperparameter and weak-reference variants. `paper_tables.py`, `trust_rates.py`, `timing.py`, `why_1n.py`, `alias_check.py` (half fraction against the full factorial): analyses | Sec. 3; App. C, D |
| `plantedmem/` | PlantedMem: `env.py` with `env_cache.pkl` (real PortBench prices, 2,828 logged LLM drafts), `world.py`, `suite.py` (every method and variant), `k_designs.py` and `k_designs16.py` (number of retrieved experiences) | Table 3, Figs. 4-6 |
| `portbench/` | `monthly_lib.py` (exact settlement, investor mandates, utility), `monthly_policies.py`, `pb_all.py` (every method, time-forward), `pb_k.py`, `detectability.py`. `data/`: every logged PortBench draft (`executions.jsonl.gz`), the retrieved experiences and their embeddings | Tables 1, 3, 4 |
| `portbench_after_cutoff/` | `eval_ext.py`: PortBench with the twelve decisions of 2025-2026 | App. D.15 |
| `portbench_prices/` | `rescore.py` and `prices.pkl`: PortBench's public price data, mandate projection and 20-session settlement | PortBench, ClassAlloc |
| `investorbench/` | `run_ib.py` (LLM runs), `ib_turnover.py` (evaluation of every method; fee on traded amounts), `ib_all.py` (shared definitions), `ib_k.py`. `data/`: public InvestorBench data; `run/`: warm-up lessons, embeddings and ExpeL insights of our run; `turnover/cache.pkl`: logged drafts | Tables 1, 3, 4 |
| `classalloc_and_robustness/` | ClassAlloc: `cb_build_data.py` (builds `classalloc/data.json` from PortBench's prices), `run_cb.py`, `cb_eval.py`, `cb_eval_extra.py`, `trusted_gain.py` (gain of the trusted decisions over the anchored action); `run_cb_delayfix.py` (feedback timing, see below); other LLMs and temperatures: `run_ib_variant.py`, `ib_eval_variant.py`, `collect_models.py`; Self-consistency with three draws per seed: `run_sc_extra.py`, `sc_eval.py`; look-ahead checks: `run_cb_masked.py`, `probe_memory.py`; `llm_client.py`; `*.sh`: the LLM runs | Tables 1, 3, 4; App. D.11-D.14 |
| `ablation/` | every ablation variant paired with MemTrial (`nt_*.py`); `nt_matrix2.py` builds Figure 4; `anchored_baselines.py`: every LLM-based baseline with MemTrial's anchored action | Fig. 4, App. D |
| `tables/`, `table1/` | mean and sd over seeds of every reported number (`sd_*.py`), Table 1 (`t1c.py`), `print_tables.py` | Tables 1, 3, Fig. 4; App. D |
| `wealth/`, `figures/`, `case_study/`, `prompts/` | wealth paths and market regimes, return and risk (`return_risk.py`), PortBench with one test month left out (`pb_months.py`); value recovery and hyperparameter data; ClassAlloc case study; prompt dump | Table 4, Figs. 3, 5, 6; App. D.6, E, G |
| `scoring.py`, `risk.py`, `mixture.py` | investor scoring, trailing covariance, exact utilities of portfolio mixtures | |

## Where the results are written

| Result | File |
|---|---|
| Table 1 | `table1/T1C_report.json` |
| Table 3 | `tables/PAPER_TABLES.md`; per seed and date: `portbench/pb_all_*.json`, `investorbench/turnover/{base,mg_MemGate}.json`, `classalloc_and_robustness/classalloc/run/eval/{base,mg0}.pkl`, `plantedmem/results/suite5_*.json`; Self-consistency: `classalloc_and_robustness/sc_runs/*/SC_EVAL.json` |
| Table 4 | `wealth/RETURN_RISK.json` (`table4`) |
| Fig. 3 (wealth) | `wealth/WEALTH_*.json`, `wealth/MARKET.json` |
| Fig. 4 (ablation) | `ablation/NT_MATRIX2.json` |
| Fig. 5 (value recovery) | `figures/VR_SUMMARY.json` |
| Fig. 6 (PlantedMem by regime) | `tables/SD_pm.json` (`util_by_regime`, `trust_by_regime`) |
| Fig. D.4 (trust-test level) | `figures/SENS_FIG.json`, `tables/SD_pb_replay_*.json`, `tables/SD_ib.json`, `tables/SD_pm.json` |
| Appendix D | `tables/SD_*.json`, `memtrial/TRUST_RATES.json`, `memtrial/sens/`, `portbench/pb_k_*.json`, `investorbench/turnover/K_IB.json`, `plantedmem/results/k*_*.json`, `portbench_after_cutoff/EXT_RESULTS.json`, `classalloc_and_robustness/classalloc/run/eval/{RESULTS,EXTRA}.json`, `memtrial/WHY_1N_pb.json`, `memtrial/TIMING.json` |
| Appendix E | `case_study/CASE_CB_balanced.json` |
| Appendix C.1 (half fraction against full factorial) | `memtrial/ALIAS_{cb,ib}.json` |
| Appendix D.5 (baselines with the anchor of MemTrial, Table D.5) | `ablation/ANCHORED.json` |
| Appendix D.6 (PortBench without one test month; the last test month) | `wealth/PB_MONTHS.json` |
| Appendix D.11 (trusted decisions on ClassAlloc) | `classalloc_and_robustness/classalloc/run/eval/TRUSTED_GAIN.json` |

## LLM runs (paid)

The runners read API keys from the environment (`OPENAI_API_KEY`; `OPENROUTER_API_KEY` for Llama, Qwen, DeepSeek, Gemini and
Claude) or from a `.env` file in this folder; no key is included. Each run saves every API response before using it, resumes
where it stopped, and stops at a hard spending cap; `--mock` runs everything end to end without network.

- InvestorBench: `cd investorbench && python3 run_ib.py`, then `python3 ib_turnover.py cache` and the steps of `reproduce.sh`.
  A new run writes `run/`, replacing our warm-up lessons there.
- ClassAlloc: `cd classalloc_and_robustness && python3 run_cb.py`, then `python3 cb_eval.py`. Each decision is scored over the 20 sessions
  after it and the holdings are carried to the next decision date; every learner, in `run_cb.py` (the FinMem-style, MemRL-style and
  Reflexion chains) and in `cb_eval.py`, uses an outcome only once its 20 sessions have passed (in 9 of the 70 test months the next
  decision comes after 19 sessions). The chains of our logged runs had received such an outcome at the next decision;
  `run_cb_delayfix.py` replayed them under the protocol above, keeping every logged decision whose prompt is unchanged and drawing the
  others anew (`classalloc/run/MANIFEST.json`, `feedback_timing`). The included `classalloc/run/eval/cache.pkl` holds the replayed run.
- Other LLMs and temperatures, Self-consistency draws, dates hidden, memorization probe: `run_all.sh`, `run_cb_models.sh`,
  `run_more_models.sh`, `run_sc_all.sh`, `run_lookahead.sh`; afterwards `refresh_all.sh` evaluates them (App. D.12-D.14).
- PortBench: the drafts come from PortBench's own S1-S5 pipeline (gpt-4.1-mini, temperature 0.7) with the memory contents
  shown in Appendix G. Our wrapper around that pipeline is not
  included; `portbench/data/` contains every logged PortBench draft, so all PortBench numbers can be recomputed.

## Not included

The raw API logs (prompts and responses) and the logs of the runs with other LLMs and temperatures, with dates hidden, and of
the memorization probe. Their evaluation code is included (`ib_eval_variant.py`, `cb_eval.py --run`, `sc_eval.py --run`,
`collect_models.py`, `probe_memory.py --eval`), as are `prompts/dump_prompts.py` and `memtrial/why_1n.py ib`, which read the
raw logs. The logs will be released with the paper.
