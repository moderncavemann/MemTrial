#!/bin/bash
# Self-consistency with three draws per seed: two extra memory-free decisions per test date and seed for every run
# (paid; run after run_all.sh and run_cb_models.sh). PortBench's extra drafts are part of ../portbench/data.
cd "$(dirname "$0")"; mkdir -p logs
laneA () { for r in main t0.0 t0.3 t1.0 nano gpt5mini; do python3 run_sc_extra.py --bench ib --run $r > logs/sc_ib_$r.log 2>&1; echo "ib $r finished $(date)"; done
           for r in main gpt-4.1-nano gpt-5-mini; do python3 run_sc_extra.py --bench cb --run $r > logs/sc_cb_$r.log 2>&1; echo "cb $r finished $(date)"; done; }
laneB () { python3 run_sc_extra.py --bench ib --run llama70b > logs/sc_ib_llama70b.log 2>&1; echo "ib llama70b finished $(date)"
           python3 run_sc_extra.py --bench cb --run llama-3.3-70b > logs/sc_cb_llama.log 2>&1; echo "cb llama finished $(date)"; }
laneA & laneB & wait
echo "all Self-consistency runs finished $(date)"
