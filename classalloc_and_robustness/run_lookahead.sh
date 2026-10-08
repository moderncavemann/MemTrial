#!/bin/bash
# Look-ahead check (paid; resumable; every run stops at its cap):
#   lane 1  ClassAlloc with every calendar date hidden from the LLM, gpt-4.1-mini, then its Self-consistency draws
#   lane 2  the same with gpt-5-mini
#   lane 3  memorization probe of the eight LLMs (date only -> recalled returns of the five asset classes)
# Afterwards (offline):  bash refresh_all.sh ; python3 probe_memory.py --eval
cd "$(dirname "$0")"; mkdir -p logs
lane () { m=$1
  python3 run_cb_masked.py --model $m > logs/cb_masked_$m.log 2>&1; echo "cb masked $m finished $(date)"
  python3 run_cb_masked.py --model $m --sc > logs/sc_cb_masked_$m.log 2>&1; echo "cb masked $m self-consistency finished $(date)"; }
lane gpt-4.1-mini & lane gpt-5-mini &
(python3 probe_memory.py > logs/probe_memory.log 2>&1; echo "memorization probe finished $(date)") &
wait
echo "all look-ahead runs finished $(date)"
