#!/usr/bin/env python3
"""Validation-threshold sweep for LOFPO random-vs-SSL checkpoints.

This script does NOT retrain models. It:
1. Loads each saved best checkpoint.
2. Sweeps thresholds on the validation split.
3. Selects the threshold with best validation Dice.
4. Applies that selected threshold once to the held-out test split.
5. Writes per-run and random-vs-SSL summary CSV files.

Run from the repository root:
    python3 training/sweep_lofpo_thresholds.py
"""

from __future__ import annotations

import argparse
import csv
import importlib.util
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import DataLoader


def import_fine_tune_module(repo_root: Path) -> Any:
    fine_tune_path = repo_root / "training" / "fine_tune.py"
    if not fine_tune_path.exists():
        raise FileNotFoundError(f"Could not find {fine_tune_path}")

    spec = importlib.util.spec_from_file_location("fine_tune_module", fine_tune_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not import {fine_tune_path}")

    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def sweep_one_run(
    *,
    fine_tune: Any,
    heldout_fp: str,
    condition: str,
    repo_root: Path,
    checkpoints_dir: Path,
    results_dir: Path,
    batch_size: int,
    num_workers: int,
    base_channels: int,
    device: torch.device,
    thresholds: list[float],
) -> dict[str, Any]:
    split_dir = repo_root / "training" / "lofo_csvs" / f"heldout_{heldout_fp}"
    train_csv = split_dir / "train.csv"
    val_csv = split_dir / "validation.csv"
    test_csv = split_dir / "test.csv"

    for path in [train_csv, val_csv, test_csv]:
        if not path.exists():
            raise FileNotFoundError(path)

    checkpoint_path = checkpoints_dir / f"lofo_{heldout_fp}_{condition}_best.pt"
    if not checkpoint_path.exists():
        raise FileNotFoundError(checkpoint_path)

    pin_memory = device.type == "cuda"

    val_dataset = fine_tune.FloodTileDataset(val_csv)
    test_dataset = fine_tune.FloodTileDataset(test_csv)

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )

    model = fine_tune.UNet(
        in_channels=3,
        out_channels=1,
        base_channels=base_channels,
    ).to(device)

    checkpoint = torch.load(
        checkpoint_path,
        map_location=device,
        weights_only=False,
    )
    model.load_state_dict(checkpoint["model_state_dict"])

    loss_fn = fine_tune.FocalDiceLoss()

    val_rows: list[dict[str, Any]] = []
    best_val_row: dict[str, Any] | None = None

    for threshold in thresholds:
        val_metrics = fine_tune.run_epoch(
            model,
            val_loader,
            loss_fn,
            device,
            optimizer=None,
            threshold=threshold,
        )

        row = {
            "heldout_fp": heldout_fp,
            "condition": condition,
            "threshold": threshold,
            "val_loss": float(val_metrics["loss"]),
            "val_dice": float(val_metrics["dice"]),
            "val_iou": float(val_metrics["iou"]),
            "val_precision": float(val_metrics["precision"]),
            "val_recall": float(val_metrics["recall"]),
            "val_accuracy": float(val_metrics["accuracy"]),
            "val_tp": int(float(val_metrics["tp"])),
            "val_fp": int(float(val_metrics["fp"])),
            "val_fn": int(float(val_metrics["fn"])),
            "val_tn": int(float(val_metrics["tn"])),
        }
        val_rows.append(row)

        if (
            best_val_row is None
            or row["val_dice"] > best_val_row["val_dice"]
            or (
                row["val_dice"] == best_val_row["val_dice"]
                and row["val_iou"] > best_val_row["val_iou"]
            )
        ):
            best_val_row = row

    assert best_val_row is not None
    selected_threshold = float(best_val_row["threshold"])

    test_metrics = fine_tune.run_epoch(
        model,
        test_loader,
        loss_fn,
        device,
        optimizer=None,
        threshold=selected_threshold,
    )

    sweep_path = results_dir / f"lofo_{heldout_fp}_{condition}_threshold_sweep.csv"
    with sweep_path.open("w", newline="", encoding="utf-8") as handle:
        fieldnames = list(val_rows[0].keys())
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(val_rows)

    return {
        "heldout_fp": heldout_fp,
        "condition": condition,
        "checkpoint_path": checkpoint_path.as_posix(),
        "train_csv": train_csv.as_posix(),
        "val_csv": val_csv.as_posix(),
        "test_csv": test_csv.as_posix(),
        "best_epoch": checkpoint.get("epoch", ""),
        "best_val_loss": checkpoint.get("best_val_loss", ""),
        "selected_threshold": selected_threshold,
        "selected_val_dice": best_val_row["val_dice"],
        "selected_val_iou": best_val_row["val_iou"],
        "selected_val_precision": best_val_row["val_precision"],
        "selected_val_recall": best_val_row["val_recall"],
        "test_loss": float(test_metrics["loss"]),
        "test_dice": float(test_metrics["dice"]),
        "test_iou": float(test_metrics["iou"]),
        "test_precision": float(test_metrics["precision"]),
        "test_recall": float(test_metrics["recall"]),
        "test_accuracy": float(test_metrics["accuracy"]),
        "test_tp": int(float(test_metrics["tp"])),
        "test_fp": int(float(test_metrics["fp"])),
        "test_fn": int(float(test_metrics["fn"])),
        "test_tn": int(float(test_metrics["tn"])),
        "sweep_csv": sweep_path.as_posix(),
    }


def write_rows(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise ValueError("No rows to write.")
    with path.open("w", newline="", encoding="utf-8") as handle:
        fieldnames = list(rows[0].keys())
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def build_pair_summary(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    by_key = {(row["heldout_fp"], row["condition"]): row for row in rows}
    summary_rows: list[dict[str, Any]] = []

    for fp_num in range(1, 8):
        heldout_fp = f"fp{fp_num}"
        random_row = by_key[(heldout_fp, "random")]
        ssl_row = by_key[(heldout_fp, "ssl")]

        summary_rows.append(
            {
                "heldout_fp": heldout_fp,
                "random_selected_threshold": random_row["selected_threshold"],
                "ssl_selected_threshold": ssl_row["selected_threshold"],
                "random_val_dice_at_selected_threshold": random_row["selected_val_dice"],
                "ssl_val_dice_at_selected_threshold": ssl_row["selected_val_dice"],
                "random_test_loss": random_row["test_loss"],
                "ssl_test_loss": ssl_row["test_loss"],
                "delta_loss_ssl_minus_random": ssl_row["test_loss"] - random_row["test_loss"],
                "random_test_dice": random_row["test_dice"],
                "ssl_test_dice": ssl_row["test_dice"],
                "delta_dice_ssl_minus_random": ssl_row["test_dice"] - random_row["test_dice"],
                "random_test_iou": random_row["test_iou"],
                "ssl_test_iou": ssl_row["test_iou"],
                "delta_iou_ssl_minus_random": ssl_row["test_iou"] - random_row["test_iou"],
                "random_precision": random_row["test_precision"],
                "ssl_precision": ssl_row["test_precision"],
                "delta_precision_ssl_minus_random": ssl_row["test_precision"] - random_row["test_precision"],
                "random_recall": random_row["test_recall"],
                "ssl_recall": ssl_row["test_recall"],
                "delta_recall_ssl_minus_random": ssl_row["test_recall"] - random_row["test_recall"],
                "random_accuracy": random_row["test_accuracy"],
                "ssl_accuracy": ssl_row["test_accuracy"],
                "delta_accuracy_ssl_minus_random": ssl_row["test_accuracy"] - random_row["test_accuracy"],
                "random_tp": random_row["test_tp"],
                "random_fp": random_row["test_fp"],
                "random_fn": random_row["test_fn"],
                "random_tn": random_row["test_tn"],
                "ssl_tp": ssl_row["test_tp"],
                "ssl_fp": ssl_row["test_fp"],
                "ssl_fn": ssl_row["test_fn"],
                "ssl_tn": ssl_row["test_tn"],
                "random_best_epoch": random_row["best_epoch"],
                "ssl_best_epoch": ssl_row["best_epoch"],
                "random_best_val_loss": random_row["best_val_loss"],
                "ssl_best_val_loss": ssl_row["best_val_loss"],
            }
        )

    return summary_rows


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Sweep validation thresholds for LOFPO random and SSL checkpoints."
    )
    parser.add_argument("--repo-root", type=Path, default=Path("."))
    parser.add_argument("--checkpoints-dir", type=Path, default=Path("models"))
    parser.add_argument("--results-dir", type=Path, default=Path("results"))
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--base-channels", type=int, default=32)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--threshold-start", type=float, default=0.05)
    parser.add_argument("--threshold-stop", type=float, default=0.95)
    parser.add_argument("--threshold-step", type=float, default=0.01)
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    repo_root = args.repo_root.resolve()
    checkpoints_dir = (repo_root / args.checkpoints_dir).resolve()
    results_dir = (repo_root / args.results_dir).resolve()
    results_dir.mkdir(parents=True, exist_ok=True)

    fine_tune = import_fine_tune_module(repo_root)
    device = torch.device(args.device)

    thresholds = []
    current = args.threshold_start
    while current <= args.threshold_stop + 1e-9:
        thresholds.append(round(current, 4))
        current += args.threshold_step

    print(f"Using device: {device}")
    print(f"Thresholds: {thresholds[0]} to {thresholds[-1]} step {args.threshold_step}")
    print(f"Total thresholds: {len(thresholds)}")

    rows: list[dict[str, Any]] = []

    for fp_num in range(1, 8):
        heldout_fp = f"fp{fp_num}"
        for condition in ["random", "ssl"]:
            print(f"===== Sweeping {heldout_fp} {condition} =====")
            row = sweep_one_run(
                fine_tune=fine_tune,
                heldout_fp=heldout_fp,
                condition=condition,
                repo_root=repo_root,
                checkpoints_dir=checkpoints_dir,
                results_dir=results_dir,
                batch_size=args.batch_size,
                num_workers=args.num_workers,
                base_channels=args.base_channels,
                device=device,
                thresholds=thresholds,
            )
            rows.append(row)
            print(
                f"{heldout_fp} {condition}: "
                f"threshold={row['selected_threshold']:.2f} "
                f"val_dice={row['selected_val_dice']:.4f} "
                f"test_dice={row['test_dice']:.4f} "
                f"test_iou={row['test_iou']:.4f} "
                f"precision={row['test_precision']:.4f} "
                f"recall={row['test_recall']:.4f}"
            )

    all_runs_path = results_dir / "lofo_threshold_selected_all_runs.csv"
    write_rows(all_runs_path, rows)

    pair_summary = build_pair_summary(rows)
    pair_summary_path = results_dir / "lofo_threshold_selected_ssl_vs_random_summary.csv"
    write_rows(pair_summary_path, pair_summary)

    mean_random_dice = sum(row["random_test_dice"] for row in pair_summary) / len(pair_summary)
    mean_ssl_dice = sum(row["ssl_test_dice"] for row in pair_summary) / len(pair_summary)
    mean_random_iou = sum(row["random_test_iou"] for row in pair_summary) / len(pair_summary)
    mean_ssl_iou = sum(row["ssl_test_iou"] for row in pair_summary) / len(pair_summary)
    dice_wins = sum(row["delta_dice_ssl_minus_random"] > 0 for row in pair_summary)
    iou_wins = sum(row["delta_iou_ssl_minus_random"] > 0 for row in pair_summary)

    print()
    print(f"Wrote {all_runs_path}")
    print(f"Wrote {pair_summary_path}")
    print()
    print("===== Threshold-selected Dice deltas =====")
    for row in pair_summary:
        print(
            row["heldout_fp"],
            "random_thr=", f"{row['random_selected_threshold']:.2f}",
            "ssl_thr=", f"{row['ssl_selected_threshold']:.2f}",
            "random_dice=", f"{row['random_test_dice']:.4f}",
            "ssl_dice=", f"{row['ssl_test_dice']:.4f}",
            "delta=", f"{row['delta_dice_ssl_minus_random']:+.4f}",
        )

    print()
    print("===== Threshold-selected summary =====")
    print(f"mean_random_dice={mean_random_dice:.4f}")
    print(f"mean_ssl_dice={mean_ssl_dice:.4f}")
    print(f"mean_delta_dice={mean_ssl_dice - mean_random_dice:+.4f}")
    print(f"ssl_dice_wins={dice_wins}/7")
    print(f"mean_random_iou={mean_random_iou:.4f}")
    print(f"mean_ssl_iou={mean_ssl_iou:.4f}")
    print(f"mean_delta_iou={mean_ssl_iou - mean_random_iou:+.4f}")
    print(f"ssl_iou_wins={iou_wins}/7")


if __name__ == "__main__":
    main()
