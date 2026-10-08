#!/usr/bin/env bash
# LLM runs of the supplementary experiments (paid; needs API keys, see ../README.md). Every run has its own hard cap
# and resumes where it stopped (every API response is saved before use), so the script can simply be started again.
#   lane A (gpt-4.1-mini): ClassAlloc, InvestorBench at temperature 0, 0.3 and 1.0
#   lane B (other LLMs):  InvestorBench with gpt-4.1-nano, gpt-5-mini and Llama-3.3-70B (OpenRouter)
set -u
cd "$(dirname "$0")"
mkdir -p logs
laneA () {
  python3 run_cb.py > logs/classalloc.log 2>&1
  for v in t0.0 t0.3 t1.0; do python3 run_ib_variant.py --variant "$v" > "logs/ib_$v.log" 2>&1; done
}
laneB () {
  for v in nano gpt5mini llama70b; do python3 run_ib_variant.py --variant "$v" > "logs/ib_$v.log" 2>&1; done
}
laneA & A=$!
laneB & B=$!
wait $A $B
echo "all runs finished $(date)"
