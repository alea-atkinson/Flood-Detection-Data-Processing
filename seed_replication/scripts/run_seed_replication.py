#!/usr/bin/env python3

"""
Run downstream seed replication for LOFPO random vs weak SimSiam.

Purpose:
- Compare random initialization vs weak SimSiam initialization.
- Use folds fp1-fp7.
- Use seeds 1, 2, 3.
- Save all replicated outputs under seed_replication/.
- Do not overwrite original six-method SSL screening outputs.

Default behavior:
- smoke mode uses 1 epoch unless --epochs is provided.
- single/full modes use 20 epochs unless --epochs is provided.
- completed runs are skipped unless --force is used.

Usage:

Dry-run smoke test:
    python3 seed_replication/scripts/run_seed_replication.py --mode smoke --dry-run

Smoke test, 1 epoch:
    python3 seed_replication/scripts/run_seed_replication.py --mode smoke

Smoke test, full 20 epochs:
    python3 seed_replication/scripts/run_seed_replication.py --mode smoke --epochs 20

One run:
    python3 seed_replication/scripts/run_seed_replication.py --mode single --fold fp1 --method random --seed 1

One run, overwrite existing:
    python3 seed_replication/scripts/run_seed_replication.py --mode single --fold fp1 --method random --seed 1 --force

Full dry run:
    python3 seed_replication/scripts/run_seed_replication.py --mode full --dry-run

Full run:
    python3 seed_replication/scripts/run_seed_replication.py --mode full
"""

import argparse
import csv
import subprocess
import sys
from datetime import datetime
from pathlib import Path


REPO_ROOT = Path("/mnt/linuxlab/home/reuuzheng/Flood-Detection-Data-Processing")

SEED_ROOT = REPO_ROOT / "seed_replication"
RAW_RESULTS_DIR = SEED_ROOT / "raw_results"
MODELS_DIR = SEED_ROOT / "models"
LOGS_DIR = SEED_ROOT / "logs"
SUMMARIES_DIR = SEED_ROOT / "summaries"

FOLDS = ["fp1", "fp2", "fp3", "fp4", "fp5", "fp6", "fp7"]
METHODS = ["random", "weak_simsiam"]
SEEDS = [1, 2, 3]

WEAK_SIMSIAM_CHECKPOINT = (
    REPO_ROOT / "training" / "pretrain_weights" / "best_weak_simsiam_ssl.pth"
)


def ensure_dirs() -> None:
    for path in [RAW_RESULTS_DIR, MODELS_DIR, LOGS_DIR, SUMMARIES_DIR]:
        path.mkdir(parents=True, exist_ok=True)


def split_dir_for_fold(fold: str) -> Path:
    return REPO_ROOT / "training" / "lofo_csvs" / f"heldout_{fold}"


def split_csvs_for_fold(fold: str) -> dict[str, Path]:
    split_dir = split_dir_for_fold(fold)

    return {
        "train": split_dir / "train.csv",
        "val": split_dir / "validation.csv",
        "test": split_dir / "test.csv",
    }


def check_required_files(fold: str, method: str) -> None:
    csvs = split_csvs_for_fold(fold)

    required = [
        REPO_ROOT / "training" / "fine_tune.py",
        csvs["train"],
        csvs["val"],
        csvs["test"],
    ]

    if method == "weak_simsiam":
        required.append(WEAK_SIMSIAM_CHECKPOINT)

    missing = [str(path) for path in required if not path.exists()]

    if missing:
        raise FileNotFoundError(
            "Missing required files:\n" + "\n".join(missing)
        )


def run_name(fold: str, method: str, seed: int) -> str:
    return f"seedrep_{fold}_{method}_seed{seed}"


def expected_outputs(name: str) -> dict[str, Path]:
    return {
        "test_metrics": RAW_RESULTS_DIR / f"{name}_test_metrics.csv",
        "history": RAW_RESULTS_DIR / f"{name}_metrics.csv",
        "model": MODELS_DIR / f"{name}_best.pt",
    }


def run_is_complete(name: str) -> bool:
    outputs = expected_outputs(name)
    return (
        outputs["test_metrics"].exists()
        and outputs["history"].exists()
        and outputs["model"].exists()
    )


def build_command(fold: str, method: str, seed: int, epochs: int) -> list[str]:
    csvs = split_csvs_for_fold(fold)
    name = run_name(fold, method, seed)

    cmd = [
        sys.executable,
        str(REPO_ROOT / "training" / "fine_tune.py"),

        "--train-csv", str(csvs["train"]),
        "--val-csv", str(csvs["val"]),
        "--test-csv", str(csvs["test"]),

        "--models-dir", str(MODELS_DIR),
        "--results-dir", str(RAW_RESULTS_DIR),
        "--run-name", name,

        "--seed", str(seed),
        "--epochs", str(epochs),
        "--freeze-epochs", "0",
        "--encoder-lr-scale", "1.0",
        "--threshold", "0.5",

        # Safer on HPCL/shared systems.
        "--num-workers", "0",
    ]

    if method == "weak_simsiam":
        cmd += ["--pretrained-checkpoint", str(WEAK_SIMSIAM_CHECKPOINT)]

    return cmd


def run_one(
    fold: str,
    method: str,
    seed: int,
    epochs: int,
    dry_run: bool = False,
    force: bool = False,
) -> dict:
    ensure_dirs()
    check_required_files(fold, method)

    name = run_name(fold, method, seed)
    log_path = LOGS_DIR / f"{name}.log"
    outputs = expected_outputs(name)
    cmd = build_command(fold, method, seed, epochs)

    already_complete = run_is_complete(name)

    row = {
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "run_name": name,
        "fold": fold,
        "method": method,
        "seed": seed,
        "epochs": epochs,
        "log_path": str(log_path),
        "test_metrics_path": str(outputs["test_metrics"]),
        "history_path": str(outputs["history"]),
        "model_path": str(outputs["model"]),
        "command": " ".join(cmd),
    }

    print("\n" + "=" * 80)
    print(f"Run: {name}")
    print(f"Epochs: {epochs}")
    print(f"Already complete: {already_complete}")
    print(f"Force overwrite: {force}")
    print("Command:")
    print(" ".join(cmd))
    print(f"Log: {log_path}")
    print("=" * 80)

    if dry_run:
        row.update({
            "status": "dry_run",
            "returncode": "",
            "test_metrics_found": outputs["test_metrics"].exists(),
            "history_found": outputs["history"].exists(),
            "model_found": outputs["model"].exists(),
        })
        return row

    if already_complete and not force:
        print(f"Skipping completed run: {name}")
        row.update({
            "status": "skipped_existing",
            "returncode": "",
            "test_metrics_found": True,
            "history_found": True,
            "model_found": True,
        })
        return row

    with log_path.open("w") as log_file:
        result = subprocess.run(
            cmd,
            cwd=REPO_ROOT,
            stdout=log_file,
            stderr=subprocess.STDOUT,
            text=True,
        )

    status = "success" if result.returncode == 0 else "failed"

    test_metrics_found = outputs["test_metrics"].exists()
    history_found = outputs["history"].exists()
    model_found = outputs["model"].exists()

    row.update({
        "status": status,
        "returncode": result.returncode,
        "test_metrics_found": test_metrics_found,
        "history_found": history_found,
        "model_found": model_found,
    })

    print(f"Finished {name}: {status}")
    print(f"Test metrics found: {test_metrics_found}")
    print(f"Training history found: {history_found}")
    print(f"Model found: {model_found}")

    if result.returncode != 0:
        print(f"Check log: {log_path}")

    return row


def write_manifest(rows: list[dict], filename: str) -> None:
    ensure_dirs()

    if not rows:
        return

    path = SUMMARIES_DIR / filename
    fieldnames = sorted(set().union(*(row.keys() for row in rows)))

    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print(f"\nWrote manifest: {path}")


def planned_runs(
    mode: str,
    fold: str | None,
    method: str | None,
    seed: int | None,
) -> list[tuple[str, str, int]]:
    if mode == "smoke":
        return [
            ("fp1", "random", 1),
            ("fp1", "weak_simsiam", 1),
        ]

    if mode == "single":
        if fold is None or method is None or seed is None:
            raise ValueError("--mode single requires --fold, --method, and --seed")
        return [(fold, method, seed)]

    if mode == "full":
        return [
            (fold_name, method_name, seed_value)
            for fold_name in FOLDS
            for method_name in METHODS
            for seed_value in SEEDS
        ]

    raise ValueError(f"Unknown mode: {mode}")


def resolve_epochs(mode: str, requested_epochs: int | None) -> int:
    if requested_epochs is not None:
        return requested_epochs

    if mode == "smoke":
        return 1

    return 20


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["smoke", "single", "full"], required=True)
    parser.add_argument("--fold", choices=FOLDS)
    parser.add_argument("--method", choices=METHODS)
    parser.add_argument("--seed", type=int, choices=SEEDS)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument(
        "--force",
        action="store_true",
        help="Overwrite/re-run completed outputs. By default, completed runs are skipped.",
    )

    args = parser.parse_args()

    ensure_dirs()

    epochs = resolve_epochs(args.mode, args.epochs)
    runs = planned_runs(args.mode, args.fold, args.method, args.seed)

    print(f"Repo root: {REPO_ROOT}")
    print(f"Seed replication root: {SEED_ROOT}")
    print(f"Mode: {args.mode}")
    print(f"Epochs: {epochs}")
    print(f"Dry run: {args.dry_run}")
    print(f"Force overwrite: {args.force}")
    print(f"Planned runs: {len(runs)}")

    rows = []

    for fold, method, seed in runs:
        row = run_one(
            fold=fold,
            method=method,
            seed=seed,
            epochs=epochs,
            dry_run=args.dry_run,
            force=args.force,
        )
        rows.append(row)

        if row.get("status") == "failed":
            print("\nStopping because a run failed.")
            break

    manifest_name = {
        "smoke": "seed_replication_smoke_manifest.csv",
        "single": "seed_replication_single_manifest.csv",
        "full": "seed_replication_full_manifest.csv",
    }[args.mode]

    if args.dry_run:
        manifest_name = manifest_name.replace(".csv", "_dry_run.csv")

    write_manifest(rows, manifest_name)


if __name__ == "__main__":
    main()
