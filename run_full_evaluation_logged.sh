#!/usr/bin/env bash
set -euo pipefail

RUN_DIR="data/eval_run_2026-02-26"
mkdir -p "$RUN_DIR"
LOG="$RUN_DIR/full_log.txt"

# start fresh log
: > "$LOG"

echo "Run started: $(date)" | tee -a "$LOG"
python --version | tee -a "$LOG"

for f in webapp/annotated/*_fingered.musicxml
do
  base=$(basename "$f" _fingered.musicxml)

  python evaluate_extract_ground_truth.py "$f" data/${base}_gt.csv
  python remove_fingerings.py "$f" data/${base}_input.musicxml

  python predict_from_mxl.py data/${base}_input.musicxml

  python evaluate_compare.py \
    data/${base}_gt.csv \
    data/predictions/${base}_input_predictions.csv

  echo "--------------------------------------"
done
