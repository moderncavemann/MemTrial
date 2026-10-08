#!/bin/bash
# InvestorBench and ClassAlloc with Gemini 2.5 Flash, Qwen3-235B, DeepSeek-V3.1 and Claude Haiku 4.5 through OpenRouter,
# with the extra memory-free draws of Self-consistency (paid; run after run_cb_models.sh and run_sc_all.sh).
# Every run is resumable (saved calls are replayed at no cost) and stops at its cap.
cd "$(dirname "$0")"; mkdir -p logs
IB="gemini25flash qwen3 deepseekv31 haiku45"; CB="gemini-2.5-flash qwen3-235b deepseek-v3.1 claude-haiku-4.5"
for v in $IB; do   # one real call per model (< USD 0.01): model id, parameters, cost per call
  if ! python3 run_ib_variant.py --variant $v --probe > logs/probe_ib_$v.log 2>&1; then echo "probe $v failed:"; tail -5 logs/probe_ib_$v.log; exit 1; fi
  tail -1 logs/probe_ib_$v.log
done
lane1 () { for v in $IB; do
  python3 run_ib_variant.py --variant $v > logs/ib_$v.log 2>&1; echo "ib $v finished $(date)"
  cap=1.5; [ "$v" = "haiku45" ] && cap=3; python3 run_sc_extra.py --bench ib --run $v --cap $cap > logs/sc_ib_$v.log 2>&1; echo "ib $v self-consistency finished $(date)"
done; }
lane2 () { for m in $CB; do
  python3 run_cb.py --model $m --cap 6 > logs/cb_$m.log 2>&1; echo "cb $m finished $(date)"
  cap=1; [ "$m" = "claude-haiku-4.5" ] && cap=2; python3 run_sc_extra.py --bench cb --run $m --cap $cap > logs/sc_cb_$m.log 2>&1; echo "cb $m self-consistency finished $(date)"
done; }
lane1 & lane2 & wait
echo "all runs finished $(date)"
