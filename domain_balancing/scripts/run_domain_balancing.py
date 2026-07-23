#!/usr/bin/env python3
"""
Run one-seed domain/flood balancing experiments.

Design:
- 7 strict LOFPO folds
- 4 sampling strategies
- random initialization U-Net only
- 20 epochs by default

Total full runs:
7 folds x 4 strategies = 28 runs
"""

from __future__ import annotations

import argparse
import csv
import subprocess
import sys
from pathlib import Path


REPO_ROOT = Path("/mnt/linuxlab/home/reuuzheng/Flood-Detection-Data-Processing")
FINE_TUNE = REPO_ROOT / "training" / "fine_tune.py"
LOFO_ROOT = REPO_ROOT / "training" / "lofo_csvs"

MODELS_DIR = REPO_ROOT / "domain_balancing" / "models"
RESULTS_DIR = REPO_ROOT / "domain_balancing" / "raw_results"
LOGS_DIR = REPO_ROOT / "domain_balancing" / "logs"
SUMMARY_DIR = REPO_ROOT / "domain_balancing" / "summaries"

FOLDS = ["fp1", "fp2", "fp3", "fp4", "fp5", "fp6", "fp7"]

SAMPLING_STRATEGIES = [
    "standard",
    "source_fp",
    "flood_bin",
    "source_fp_x_flood_bin",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Launch domain/flood balancing LOFPO experiments."
    )
    parser.add_argument(
        "--mode",
        choices=["smoke", "single", "full"],
        default="single",
        help=(
            "smoke = one fp1 standard run for 1 epoch unless --epochs is set; "
            "single = one fold/strategy; "
            "full = all 28 runs."
        ),
    )
    parser.add_argument(
        "--fold",
        choices=FOLDS,
        default="fp1",
        help="Fold to run in single mode.",
    )
    parser.add_argument(
        "--sampling-strategy",
        choices=SAMPLING_STRATEGIES,
        default="standard",
        help="Sampling strategy to run in single mode.",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=None,
        help="Number of training epochs. Defaults: smoke=1, single/full=20.",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--force", action="store_true", help="Rerun existing outputs.")
    parser.add_argument(
        "--sampler-max-weight-multiplier",
        type=float,
        default=5.0,
        help="Weight clipping multiplier for balanced samplers.",
    )
    return parser.parse_args()


def build_plan(args: argparse.Namespace) -> list[tuple[str, str]]:
    if args.mode == "smoke":
        return [("fp1", "standard")]

    if args.mode == "single":
        return [(args.fold, args.sampling_strategy)]

    return [
        (fold, strategy)
        for fold in FOLDS
        for strategy in SAMPLING_STRATEGIES
    ]


def default_epochs(args: argparse.Namespace) -> int:
    if args.epochs is not None:
        return args.epochs
    if args.mode == "smoke":
        return 1
    return 20


def run_one(
    fold: str,
    strategy: str,
    args: argparse.Namespace,
    epochs: int,
) -> dict[str, str | int]:
    heldout_dir = LOFO_ROOT / f"heldout_{fold}"

    train_csv = heldout_dir / "train.csv"
    val_csv = heldout_dir / "validation.csv"
    test_csv = heldout_dir / "test.csv"

    run_name = f"domain_balance_{strategy}_{fold}_seed{args.seed}"

    test_metrics = RESULTS_DIR / f"{run_name}_test_metrics.csv"
    metrics = RESULTS_DIR / f"{run_name}_metrics.csv"
    checkpoint = MODELS_DIR / f"{run_name}_best.pt"
    log_path = LOGS_DIR / f"{run_name}.log"

    if test_metrics.exists() and checkpoint.exists() and not args.force:
        print(f"[SKIP] {run_name} already has checkpoint and test metrics.")
        return {
            "run_name": run_name,
            "fold": fold,
            "sampling_strategy": strategy,
            "status": "skipped_existing",
            "returncode": 0,
            "log_path": log_path.as_posix(),
            "test_metrics": test_metrics.as_posix(),
        }

    cmd = [
        sys.executable,
        str(FINE_TUNE),
        "--train-csv",
        str(train_csv),
        "--val-csv",
        str(val_csv),
        "--test-csv",
        str(test_csv),
        "--epochs",
        str(epochs),
        "--batch-size",
        str(args.batch_size),
        "--num-workers",
        str(args.num_workers),
        "--models-dir",
        str(MODELS_DIR),
        "--results-dir",
        str(RESULTS_DIR),
        "--run-name",
        run_name,
        "--device",
        args.device,
        "--seed",
        str(args.seed),
        "--freeze-epochs",
        "0",
        "--encoder-lr-scale",
        "1.0",
        "--sampling-strategy",
        strategy,
        "--sampler-max-weight-multiplier",
        str(args.sampler_max_weight_multiplier),
    ]

    print(f"[RUN] {run_name}")
    print(" ".join(cmd))

    LOGS_DIR.mkdir(parents=True, exist_ok=True)
    with log_path.open("w", encoding="utf-8") as log_file:
        proc = subprocess.run(
            cmd,
            cwd=REPO_ROOT,
            stdout=log_file,
            stderr=subprocess.STDOUT,
            text=True,
        )

    status = "success" if proc.returncode == 0 else "failed"

    print(f"[{status.upper()}] {run_name} returncode={proc.returncode}")
    if status == "failed":
        print(f"  See log: {log_path}")

    return {
        "run_name": run_name,
        "fold": fold,
        "sampling_strategy": strategy,
        "status": status,
        "returncode": proc.returncode,
        "log_path": log_path.as_posix(),
        "metrics": metrics.as_posix(),
        "test_metrics": test_metrics.as_posix(),
        "checkpoint": checkpoint.as_posix(),
    }


def write_manifest(rows: list[dict[str, str | int]], mode: str) -> Path:
    SUMMARY_DIR.mkdir(parents=True, exist_ok=True)
    manifest_path = SUMMARY_DIR / f"domain_balancing_{mode}_manifest.csv"

    fieldnames = [
        "run_name",
        "fold",
        "sampling_strategy",
        "status",
        "returncode",
        "log_path",
        "metrics",
        "test_metrics",
        "checkpoint",
    ]

    with manifest_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in fieldnames})

    return manifest_path


def main() -> None:
    args = parse_args()
    epochs = default_epochs(args)

    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    LOGS_DIR.mkdir(parents=True, exist_ok=True)
    SUMMARY_DIR.mkdir(parents=True, exist_ok=True)

    plan = build_plan(args)

    print("Domain balancing launcher")
    print("=========================")
    print(f"Mode: {args.mode}")
    print(f"Epochs: {epochs}")
    print(f"Seed: {args.seed}")
    print(f"Planned runs: {len(plan)}")
    print()

    rows = []
    for fold, strategy in plan:
        row = run_one(fold, strategy, args, epochs)
        rows.append(row)

    manifest_path = write_manifest(rows, args.mode)

    n_success = sum(row["status"] == "success" for row in rows)
    n_skipped = sum(row["status"] == "skipped_existing" for row in rows)
    n_failed = sum(row["status"] == "failed" for row in rows)

    print()
    print("Done.")
    print(f"Manifest: {manifest_path}")
    print(f"Success: {n_success}")
    print(f"Skipped existing: {n_skipped}")
    print(f"Failed: {n_failed}")

    if n_failed > 0:
        raise SystemExit(1)


if __name__ == "__main__":
    main()