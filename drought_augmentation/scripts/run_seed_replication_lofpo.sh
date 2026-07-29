#!/usr/bin/env bash
set -euo pipefail

SPLIT_ROOT="training/fine_tune_csvs/no_overlap_converted"
PRETRAINED_WEIGHTS="drought_augmentation/models/drought_negative_pretraining_best.pt"

MODELS_DIR="drought_augmentation/models_seed_replication"
RESULTS_DIR="drought_augmentation/raw_results_seed_replication"
LOG_DIR="drought_augmentation/logs_seed_replication"

EPOCHS=20
BATCH_SIZE=16
NUM_WORKERS=0

mkdir -p "$MODELS_DIR" "$RESULTS_DIR" "$LOG_DIR"

for SEED in 42 123 999; do
  for FP in 1 2 3 4 5 6 7; do

    echo "============================================================"
    echo "Random init | seed ${SEED} | fp${FP}"
    echo "============================================================"

    python3 training/milton_pretraining.py \
      --train-csv "${SPLIT_ROOT}/held_out_fp${FP}_train.csv" \
      --val-csv "${SPLIT_ROOT}/held_out_fp${FP}_val.csv" \
      --test-csv "${SPLIT_ROOT}/held_out_fp${FP}_test.csv" \
      --epochs "$EPOCHS" \
      --batch-size "$BATCH_SIZE" \
      --num-workers "$NUM_WORKERS" \
      --seed "$SEED" \
      --models-dir "$MODELS_DIR" \
      --results-dir "$RESULTS_DIR" \
      --run-name "fp${FP}_seed${SEED}_random_init" \
      2>&1 | tee "${LOG_DIR}/fp${FP}_seed${SEED}_random_init.log"

    echo "============================================================"
    echo "Drought-negative init | seed ${SEED} | fp${FP}"
    echo "============================================================"

    python3 training/milton_pretraining.py \
      --pretrained_weights "$PRETRAINED_WEIGHTS" \
      --train-csv "${SPLIT_ROOT}/held_out_fp${FP}_train.csv" \
      --val-csv "${SPLIT_ROOT}/held_out_fp${FP}_val.csv" \
      --test-csv "${SPLIT_ROOT}/held_out_fp${FP}_test.csv" \
      --epochs "$EPOCHS" \
      --batch-size "$BATCH_SIZE" \
      --num-workers "$NUM_WORKERS" \
      --seed "$SEED" \
      --models-dir "$MODELS_DIR" \
      --results-dir "$RESULTS_DIR" \
      --run-name "fp${FP}_seed${SEED}_drought_negative_init" \
      2>&1 | tee "${LOG_DIR}/fp${FP}_seed${SEED}_drought_negative_init.log"

  done
done
