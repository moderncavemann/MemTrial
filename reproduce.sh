#!/usr/bin/env bash
# Recompute the results of the paper from the included LLM outputs: offline, no API key, about two hours on 4 cores.
# Every step writes its outputs next to its script; see README.md for where each table and figure is.
set -euo pipefail
cd "$(dirname "$0")"
PY=${PYTHON:-python3}
CFGS="full-price raw-price"; INVS="conservative balanced aggressive"

echo "== PlantedMem (100 test worlds per regime and investor)"
(cd plantedmem
 for g in $INVS; do $PY suite.py prun $g 15 115; done                                  # results/suite5_*.json
 for g in $INVS; do $PY k_designs.py prun $g 15 65; $PY k_designs16.py prun $g 15 65; done)   # number of retrieved experiences
(cd memtrial; for g in $INVS; do $PY sens.py ct $g 15 115; done)                        # hyperparameters, weak reference
(cd ablation; $PY nt_pm.py 15 115; $PY nt_pmq.py 15 115)                                 # never trusting; fixed blend
(cd figures; $PY value_recovery.py run 1e9; $PY value_recovery.py summary)               # value recovery

echo "== PortBench"
(cd portbench
 $PY monthly_lib.py                                                                     # investor projections of every draft (resumable)
 for c in $CFGS; do for p in $INVS; do $PY pb_all.py $c $p; done; $PY pb_k.py $c; done
 $PY detectability.py)
(cd memtrial; for c in $CFGS; do for p in $INVS; do $PY sens.py pb $c $p; done; done)
(cd ablation; for c in $CFGS; do for p in $INVS; do $PY nt_pb.py $c $p; done; done)
(cd portbench_after_cutoff
 while true; do out=$($PY eval_ext.py scores 600); echo "$out"; case "$out" in *"remaining 0;"*) break;; esac; done
 for c in $CFGS; do for p in $INVS; do $PY eval_ext.py run $c $p; done; done
 $PY eval_ext.py report)

echo "== InvestorBench"
(cd investorbench
 $PY ib_turnover.py base
 $PY ib_turnover.py mt "$($PY -c 'import ib_turnover as T; print(";".join(T.VARIANTS))')"
 $PY ib_turnover.py merge; $PY ib_turnover.py diag; $PY ib_k.py)

echo "== ClassAlloc and Self-consistency with three draws per seed"
(cd classalloc_and_robustness
 $PY cb_eval.py; $PY cb_eval_extra.py
 for b in ib cb pb; do $PY sc_eval.py --bench $b; done)

echo "== Ablation table, mean +- sd tables, Table 1, figure data"
(cd ablation; $PY nt_ib.py; $PY nt_cb.py; $PY nt_fixedq_ib_cb.py; $PY nt_matrix.py; $PY nt_matrix2.py; $PY nt_summary_pm.py)
(cd tables
 $PY sd_pb.py fast; $PY sd_pb.py t1; for c in $CFGS; do $PY sd_pb.py replay $c; done
 $PY sd_ib.py; $PY sd_cb.py; $PY sd_pm.py
 $PY sd_ibk.py "k=2, full (4 drafts)" "k=3, half (4 drafts)" "k=3, full (8 drafts)" "k=4, half (8 drafts)" "k=4, full (16 drafts)")
(cd table1; for s in pb ib cb report; do $PY t1c.py $s; done)
(cd memtrial; $PY trust_rates.py; $PY why_1n.py pb; $PY timing.py)
(cd figures; $PY sens_data.py)
(cd wealth
 $PY market.py; $PY wealth_ib.py; $PY wealth_cb.py
 for c in $CFGS; do for p in $INVS; do $PY wealth_pb.py $c $p; done; done
 $PY return_risk.py                                                                      # Table 4
 $PY pb_months.py)                                                                       # PortBench months one at a time (App. D.6)
(cd case_study; $PY case_cb.py balanced 2022-03-01 2022-04-01 2022-06-01 2022-09-01 2024-09-03)
(cd classalloc_and_robustness; $PY trusted_gain.py)                                     # trusted decisions on ClassAlloc (App. D.11)
(cd ablation                                                                             # baselines with the anchor of MemTrial (App. D.5)
 for c in $CFGS; do for p in $INVS; do $PY anchored_baselines.py pb $c $p; done; done
 $PY anchored_baselines.py ib; $PY anchored_baselines.py cb; $PY anchored_baselines.py pm 15 115 4; $PY anchored_baselines.py report)
(cd memtrial; $PY alias_check.py cb; $PY alias_check.py ib)                            # half fraction vs full factorial (App. C.1)
(cd tables; $PY print_tables.py)                                                         # Tables 1, 3 and Figure 4 -> tables/PAPER_TABLES.md
echo "done"
