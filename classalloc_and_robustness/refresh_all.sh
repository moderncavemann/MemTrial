#!/bin/bash
# Offline evaluation of every finished LLM run (no API calls): ClassAlloc and InvestorBench runs with other LLMs,
# the three-draw Self-consistency, the wealth paths, the collected results (RESULTS_MODELS.json) and the run costs.
cd "$(dirname "$0")"
for d in classalloc/run_*_t0.7; do
  [ -f "$d/RUN_SUMMARY.json" ] || continue; r=$(basename "$d")
  if [ ! -f "$d/eval/RESULTS.json" ] || [ "$d/RUN_SUMMARY.json" -nt "$d/eval/RESULTS.json" ]; then
    for st in cache base mg0 mg1 mg2 summary; do python3 cb_eval.py --run "$r" --stage $st > /dev/null 2>&1 || echo "cb_eval $r $st failed"; done
    echo "evaluated $r"
  fi
  (cd ../wealth && python3 wealth_cb.py "$r" > /dev/null 2>&1 || echo "wealth_cb $r failed")
done
for d in ib_runs/*/; do
  v=$(basename "$d"); [ -f "$d/RUN_SUMMARY.json" ] || continue
  if [ ! -f "$d/eval/SUMMARY.json" ] || [ "$d/RUN_SUMMARY.json" -nt "$d/eval/SUMMARY.json" ]; then
    for st in cache base mg mg-nogate summary; do python3 ib_eval_variant.py "$v" --stage $st > /dev/null 2>&1 || echo "ib_eval $v $st failed"; done
    echo "evaluated $v"
  fi
done
for d in sc_runs/*/; do
  n=$(basename "$d"); case "$n" in *_mock) continue;; esac; [ -f "$d/RUN_SUMMARY.json" ] || continue
  b=${n%%_*}; r=${n#*_}
  if [ "$b" = "pb" ]; then continue; fi
  python3 sc_eval.py --bench "$b" --run "$r" > /dev/null 2>&1 || echo "sc_eval $n failed"
done
(cd ../wealth && python3 wealth_ib.py > /dev/null 2>&1 || echo "wealth_ib failed")
python3 collect_models.py | tail -26
python3 - <<'PY'
import json, glob, os
c = {}
for f in sorted(glob.glob("ib_runs/*/RUN_SUMMARY.json") + glob.glob("classalloc/run*/RUN_SUMMARY.json") + glob.glob("sc_runs/*/RUN_SUMMARY.json")):
    if "mock" in f: continue
    d = json.load(open(f)); c[os.path.dirname(f)] = round(d.get("spent_usd_conservative", d.get("spent_usd", 0)) or 0, 3)
json.dump(c, open("COSTS.json", "w"), indent=1); print("costs", c)
PY
