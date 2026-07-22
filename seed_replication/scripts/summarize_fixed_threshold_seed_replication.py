#!/usr/bin/env python3

"""
Summarize fixed-threshold seed replication results.

Inputs:
- seed_replication/raw_results/seedrep_*_test_metrics.csv

Outputs:
- seed_replication/tables/seed_replication_fixed_threshold_all_runs.csv
- seed_replication/tables/seed_replication_fixed_threshold_summary.csv
"""

import csv
import re
from pathlib import Path
from statistics import mean, stdev


REPO_ROOT = Path("/mnt/linuxlab/home/reuuzheng/Flood-Detection-Data-Processing")
SEED_ROOT = REPO_ROOT / "seed_replication"
RAW_RESULTS_DIR = SEED_ROOT / "raw_results"
TABLES_DIR = SEED_ROOT / "tables"
TABLES_DIR.mkdir(parents=True, exist_ok=True)

ALL_RUNS_OUT = TABLES_DIR / "seed_replication_fixed_threshold_all_runs.csv"
SUMMARY_OUT = TABLES_DIR / "seed_replication_fixed_threshold_summary.csv"

RUN_RE = re.compile(r"seedrep_(fp\d+)_(random|weak_simsiam)_seed(\d+)_test_metrics\.csv")

METRIC_ALIASES = {
    "dice": ["dice", "test_dice", "global_dice", "test_global_dice"],
    "iou": ["iou", "test_iou", "global_iou", "test_global_iou"],
    "precision": ["precision", "test_precision", "global_precision", "test_global_precision"],
    "recall": ["recall", "test_recall", "global_recall", "test_global_recall"],
    "accuracy": ["accuracy", "test_accuracy", "global_accuracy", "test_global_accuracy"],
}

METRIC_COLUMNS = list(METRIC_ALIASES.keys())


def read_single_metric_csv(path: Path) -> dict:
    with path.open(newline="") as f:
        rows = list(csv.DictReader(f))

    if len(rows) != 1:
        raise ValueError(f"Expected one row in {path}, found {len(rows)}")

    return rows[0]


def get_metric(row: dict, metric: str, path: Path) -> float:
    for candidate in METRIC_ALIASES[metric]:
        if candidate in row and row[candidate] != "":
            return float(row[candidate])

    available = ", ".join(row.keys())
    raise KeyError(
        f"Could not find metric '{metric}' in {path}\n"
        f"Tried: {METRIC_ALIASES[metric]}\n"
        f"Available columns: {available}"
    )


def parse_file(path: Path) -> dict:
    match = RUN_RE.match(path.name)
    if not match:
        raise ValueError(f"Unexpected filename: {path.name}")

    fold, method, seed = match.groups()
    row = read_single_metric_csv(path)

    out = {
        "fold": fold,
        "method": method,
        "seed": int(seed),
        "run_name": f"seedrep_{fold}_{method}_seed{seed}",
        "path": str(path),
    }

    for metric in METRIC_COLUMNS:
        out[metric] = get_metric(row, metric, path)

    return out


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        raise ValueError("No rows to write")

    fieldnames = list(rows[0].keys())

    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def summarize(rows: list[dict]) -> list[dict]:
    summary_rows = []
    groups = {}

    for row in rows:
        key = (row["fold"], row["method"])
        groups.setdefault(key, []).append(row)

    for (fold, method), group_rows in sorted(groups.items()):
        out = {
            "fold": fold,
            "method": method,
            "n_seeds": len(group_rows),
        }

        for metric in METRIC_COLUMNS:
            values = [r[metric] for r in group_rows]
            out[f"{metric}_mean"] = mean(values)
            out[f"{metric}_std"] = stdev(values) if len(values) > 1 else 0.0
            out[f"{metric}_min"] = min(values)
            out[f"{metric}_max"] = max(values)

        summary_rows.append(out)

    for method in ["random", "weak_simsiam"]:
        method_rows = [r for r in rows if r["method"] == method]

        out = {
            "fold": "ALL",
            "method": method,
            "n_seeds": len(method_rows),
        }

        for metric in METRIC_COLUMNS:
            values = [r[metric] for r in method_rows]
            out[f"{metric}_mean"] = mean(values)
            out[f"{metric}_std"] = stdev(values) if len(values) > 1 else 0.0
            out[f"{metric}_min"] = min(values)
            out[f"{metric}_max"] = max(values)

        summary_rows.append(out)

    return summary_rows


def main() -> None:
    files = sorted(RAW_RESULTS_DIR.glob("seedrep_*_test_metrics.csv"))

    if len(files) != 42:
        print(f"WARNING: expected 42 test metric files, found {len(files)}")

    rows = [parse_file(path) for path in files]
    rows = sorted(rows, key=lambda r: (r["fold"], r["method"], r["seed"]))

    write_csv(ALL_RUNS_OUT, rows)

    summary_rows = summarize(rows)
    write_csv(SUMMARY_OUT, summary_rows)

    print(f"Wrote {ALL_RUNS_OUT}")
    print(f"Wrote {SUMMARY_OUT}")

    random_all = next(r for r in summary_rows if r["fold"] == "ALL" and r["method"] == "random")
    weak_all = next(r for r in summary_rows if r["fold"] == "ALL" and r["method"] == "weak_simsiam")

    print()
    print("Overall fixed-threshold seed replication:")
    print(f"Random Dice mean ± std:       {random_all['dice_mean']:.4f} ± {random_all['dice_std']:.4f}")
    print(f"Weak SimSiam Dice mean ± std: {weak_all['dice_mean']:.4f} ± {weak_all['dice_std']:.4f}")
    print(f"Difference:                   {weak_all['dice_mean'] - random_all['dice_mean']:+.4f}")
    print()
    print("Overall IoU:")
    print(f"Random IoU mean ± std:        {random_all['iou_mean']:.4f} ± {random_all['iou_std']:.4f}")
    print(f"Weak SimSiam IoU mean ± std:  {weak_all['iou_mean']:.4f} ± {weak_all['iou_std']:.4f}")
    print(f"Difference:                   {weak_all['iou_mean'] - random_all['iou_mean']:+.4f}")


if __name__ == "__main__":
    main()
