#!/usr/bin/env bash
set -euo pipefail

SPLIT_ROOT="training/fine_tune_csvs/no_overlap_converted"

MODELS_DIR="drought_augmentation/models"
RESULTS_DIR="drought_augmentation/raw_results"
LOG_DIR="drought_augmentation/logs"

EPOCHS=20
BATCH_SIZE=16
NUM_WORKERS=0

mkdir -p "$MODELS_DIR" "$RESULTS_DIR" "$LOG_DIR"

echo "Running random-initialized LOFPO baseline"
echo "Split root: $SPLIT_ROOT"
echo "Epochs: $EPOCHS"
echo "Batch size: $BATCH_SIZE"
echo "Num workers: $NUM_WORKERS"

for FP in 1 2 3 4 5 6 7; do
  echo "============================================================"
  echo "Fold fp${FP}: random initialization"
  echo "============================================================"

  python3 training/milton_pretraining.py \
    --train-csv "${SPLIT_ROOT}/held_out_fp${FP}_train.csv" \
    --val-csv "${SPLIT_ROOT}/held_out_fp${FP}_val.csv" \
    --test-csv "${SPLIT_ROOT}/held_out_fp${FP}_test.csv" \
    --epochs "$EPOCHS" \
    --batch-size "$BATCH_SIZE" \
    --num-workers "$NUM_WORKERS" \
    --models-dir "$MODELS_DIR" \
    --results-dir "$RESULTS_DIR" \
    --run-name "fp${FP}_random_init" \
    2>&1 | tee "${LOG_DIR}/fp${FP}_random_init.log"
done

echo "Finished random-initialized LOFPO baseline."