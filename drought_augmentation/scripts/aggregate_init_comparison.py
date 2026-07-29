#!/usr/bin/env python3

from pathlib import Path
import pandas as pd

METRICS_DIR = Path("drought_augmentation/handoff_for_mac/test_metrics")
OUT_DIR = Path("drought_augmentation/tables")
OUT_DIR.mkdir(parents=True, exist_ok=True)

rows = []

for condition in ["random_init", "drought_negative_init"]:
    for fp in range(1, 8):
        path = METRICS_DIR / f"fp{fp}_{condition}_test_metrics.csv"

        if not path.exists():
            raise FileNotFoundError(f"Missing expected file: {path}")

        df = pd.read_csv(path)

        if len(df) != 1:
            raise ValueError(f"Expected exactly one row in {path}, got {len(df)}")

        row = df.iloc[0].to_dict()
        row["fold"] = f"fp{fp}"
        row["condition"] = condition
        row["source_file"] = str(path)
        rows.append(row)

all_df = pd.DataFrame(rows)

ordered_cols = [
    "condition",
    "fold",
    "run_name",
    "best_epoch",
    "best_val_loss",
    "test_loss",
    "test_dice",
    "test_iou",
    "test_precision",
    "test_recall",
    "train_csv",
    "val_csv",
    "test_csv",
    "pretrained_weights",
    "source_file",
]

all_df = all_df[[c for c in ordered_cols if c in all_df.columns]]

by_fold_path = OUT_DIR / "init_comparison_by_fold.csv"
all_df.to_csv(by_fold_path, index=False)

summary = (
    all_df.groupby("condition")
    .agg(
        mean_test_dice=("test_dice", "mean"),
        std_test_dice=("test_dice", "std"),
        mean_test_iou=("test_iou", "mean"),
        std_test_iou=("test_iou", "std"),
        mean_test_precision=("test_precision", "mean"),
        std_test_precision=("test_precision", "std"),
        mean_test_recall=("test_recall", "mean"),
        std_test_recall=("test_recall", "std"),
        mean_test_loss=("test_loss", "mean"),
        mean_best_epoch=("best_epoch", "mean"),
    )
    .reset_index()
)

summary_path = OUT_DIR / "init_comparison_summary.csv"
summary.to_csv(summary_path, index=False)

pivot = all_df.pivot(
    index="fold",
    columns="condition",
    values=["test_dice", "test_iou", "test_precision", "test_recall"],
)

pivot.columns = [f"{metric}_{condition}" for metric, condition in pivot.columns]
pivot = pivot.reset_index()

for metric in ["test_dice", "test_iou", "test_precision", "test_recall"]:
    random_col = f"{metric}_random_init"
    drought_col = f"{metric}_drought_negative_init"
    delta_col = f"{metric}_delta_drought_minus_random"

    pivot[delta_col] = pivot[drought_col] - pivot[random_col]

delta_path = OUT_DIR / "init_comparison_paired_deltas.csv"
pivot.to_csv(delta_path, index=False)

print(f"Wrote {by_fold_path}")
print(f"Wrote {summary_path}")
print(f"Wrote {delta_path}")

print("\nSummary:")
print(summary.to_string(index=False))

print("\nPaired deltas:")
print(pivot.to_string(index=False))