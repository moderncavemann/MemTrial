#!/bin/bash
# ClassAlloc with gpt-4.1-nano, gpt-5-mini and Llama-3.3-70B (paid). Each run writes classalloc/run_<model>_t0.7/
# (resumable: rerunning replays saved calls at no cost) and stops at its cap.
cd "$(dirname "$0")"; mkdir -p logs
lane () { for m in "$@"; do echo "$m started $(date)"; python3 run_cb.py --model "$m" --cap 6 > "logs/cb_${m}.log" 2>&1; echo "$m finished $(date) (exit $?)"; done; }
lane gpt-4.1-nano gpt-5-mini &
lane llama-3.3-70b &
wait
echo "all ClassAlloc model runs finished $(date)"
