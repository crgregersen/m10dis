#!/usr/bin/env bash
# Score the newest benchmark run folder of a name (CPU, pueue group default), v2 (night 2026-11-04):
#   run_bench_score_v2.sh <bench name, e.g. mmlu_redux_5shot> <mmlu|lcb>
# The run folders are root-owned (the benchmark runner runs via sudo), so v1 could not write its outputs there.
# v2 reads the run folder and writes score.log / score.json / scored.jsonl to
# benchmark-results/night-runs/scores/<run folder name>/ (owned by the scoring user). Prints the score line and SCORE_DONE.
set -uo pipefail
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/m10dis_env.sh"
NAME=$1; TYPE=$2
D=$(ls -td $M10DIS_RUNS/qwen38-bench/"$NAME"-*/ 2>/dev/null | head -1)
[[ -n $D && -s $D/results.jsonl ]] || { echo "SCORE_FAIL,no results for $NAME"; exit 3; }
S=$M10DIS_RUNS/night-runs/scores/$(basename "$D")
mkdir -p "$S" || { echo "SCORE_FAIL,cannot create $S"; exit 5; }
echo "SCORE_RUN,$D,out,$S"
cd /tmp
case $TYPE in
  mmlu) /usr/bin/python3 -I $M10DIS_HOME/src/score_mmlu.py "$D/items.jsonl" "$D/results.jsonl" \
          --out "$S/score.json" > "$S/score.log" 2>&1; rc=$? ;;
  lcb)  /usr/bin/python3 -I $M10DIS_HOME/src/score_lcb.py $M10DIS_BENCH_DATA/lcb/test6.jsonl \
          "$D/results.jsonl" --out "$S/scored.jsonl" > "$S/score.log" 2>&1; rc=$? ;;
  *) echo "SCORE_FAIL,unknown type $TYPE"; exit 2 ;;
esac
cat "$S/score.log"
grep -qE '^(MMLU_SCORE|LCB_TOTAL),' "$S/score.log" || { echo "SCORE_FAIL,no score line"; exit 4; }
echo "SCORE_DONE,rc,$rc,$S"
exit $rc
