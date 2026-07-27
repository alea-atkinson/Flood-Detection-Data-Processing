#!/usr/bin/env python3
"""Sweep --encoder-lr-scale (across multiple seeds) for train_unet_finetune.py
and summarize mean/std dice and IoU per scale.

This runs the actual comparison instead of guessing from a single run --
important because an unstable LR can look great on one seed and terrible on
the next, which single-run numbers can't distinguish from a real effect.

Example:
    python3 sweep_encoder_lr_scale.py \
        --script-path train_unet_finetune.py \
        --train-csv train.csv --val-csv val.csv --test-csv test.csv \
        --pretrained-checkpoint checkpoints/best_mae.pth \
        --freeze-epochs 5 \
        --lr-scales 0.1 0.3 0.5 1.0 2.0 \
        --seeds 0 1 2 \
        --epochs 20

Every run's checkpoint/results-dir args are forwarded to
train_unet_finetune.py unchanged, so this behaves like calling that script
directly, just looped over (lr_scale, seed) combinations with a unique
--run-name per combination.

Output:
    <sweep-results-dir>/<sweep-name>_per_run.csv   -- one row per (lr_scale, seed)
    <sweep-results-dir>/<sweep-name>_summary.csv   -- mean/std per lr_scale
"""

from __future__ import annotations

import argparse
import csv
import re
import statistics
import subprocess
import sys
from pathlib import Path

TEST_METRICS_RE = re.compile(
    r"Best checkpoint test metrics: loss=([\d.eE+-]+) dice=([\d.eE+-]+) iou=([\d.eE+-]+)"
)


def best_val_metrics_from_csv(metrics_path: Path) -> dict | None:
    """Reads a run's per-epoch metrics CSV and returns the val metrics at the
    epoch with lowest val_loss -- matches the checkpoint-selection logic in
    train_unet_finetune.py, so this is the val performance of the checkpoint
    that actually got tested."""
    if not metrics_path.exists():
        return None
    with metrics_path.open("r", newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        return None
    best_row = min(rows, key=lambda r: float(r["val_loss"]))
    return {
        "val_loss": float(best_row["val_loss"]),
        "val_dice": float(best_row["val_dice"]),
        "val_iou": float(best_row["val_iou"]),
        "best_epoch": int(best_row["epoch"]),
    }


def run_one(script_path: Path, lr_scale: float, seed: int, run_name: str, passthrough_args: list[str]) -> dict:
    cmd = [
        sys.executable, str(script_path),
        "--encoder-lr-scale", str(lr_scale),
        "--seed", str(seed),
        "--run-name", run_name,
        *passthrough_args,
    ]
    print(f"\n=== Running: lr_scale={lr_scale} seed={seed} run_name={run_name} ===")
    print(" ".join(cmd))

    result = subprocess.run(cmd, capture_output=True, text=True)
    print(result.stdout[-3000:])  # tail only, in case of long logs
    if result.returncode != 0:
        print(result.stderr[-3000:])
        raise RuntimeError(f"Run failed (lr_scale={lr_scale}, seed={seed}): exit code {result.returncode}")

    match = TEST_METRICS_RE.search(result.stdout)
    if match is None:
        raise RuntimeError(
            f"Could not find 'Best checkpoint test metrics' line in output for "
            f"lr_scale={lr_scale}, seed={seed}. Check the run's stdout above."
        )
    test_loss, test_dice, test_iou = (float(x) for x in match.groups())

    return {
        "lr_scale": lr_scale,
        "seed": seed,
        "run_name": run_name,
        "test_loss": test_loss,
        "test_dice": test_dice,
        "test_iou": test_iou,
    }


def build_passthrough_args(args: argparse.Namespace) -> list[str]:
    """Forwards this script's own args to train_unet_finetune.py, skipping
    anything left unset (None) so the underlying script's own defaults apply."""
    mapping = {
        "--train-csv": args.train_csv,
        "--val-csv": args.val_csv,
        "--test-csv": args.test_csv,
        "--epochs": args.epochs,
        "--batch-size": args.batch_size,
        "--learning-rate": args.learning_rate,
        "--weight-decay": args.weight_decay,
        "--freeze-epochs": args.freeze_epochs,
        "--base-channels": args.base_channels,
        "--pretrained-checkpoint": args.pretrained_checkpoint,
        "--models-dir": args.models_dir,
        "--results-dir": args.results_dir,
        "--device": args.device,
    }
    passthrough = []
    for flag, value in mapping.items():
        if value is not None:
            passthrough.extend([flag, str(value)])
    return passthrough


def summarize(all_results: list[dict]) -> list[dict]:
    """Groups per-run results by lr_scale and computes mean/std of test and
    val dice/IoU across seeds."""
    by_scale: dict[float, list[dict]] = {}
    for r in all_results:
        by_scale.setdefault(r["lr_scale"], []).append(r)

    summary_rows = []
    for lr_scale, runs in sorted(by_scale.items()):
        test_dices = [r["test_dice"] for r in runs]
        test_ious = [r["test_iou"] for r in runs]
        val_dices = [r["val_dice"] for r in runs if "val_dice" in r]
        val_ious = [r["val_iou"] for r in runs if "val_iou" in r]

        summary_rows.append({
            "lr_scale": lr_scale,
            "n_seeds": len(runs),
            "test_dice_mean": statistics.mean(test_dices),
            "test_dice_std": statistics.pstdev(test_dices) if len(test_dices) > 1 else 0.0,
            "test_iou_mean": statistics.mean(test_ious),
            "test_iou_std": statistics.pstdev(test_ious) if len(test_ious) > 1 else 0.0,
            "val_dice_mean": statistics.mean(val_dices) if val_dices else float("nan"),
            "val_dice_std": statistics.pstdev(val_dices) if len(val_dices) > 1 else 0.0,
            "val_iou_mean": statistics.mean(val_ious) if val_ious else float("nan"),
            "val_iou_std": statistics.pstdev(val_ious) if len(val_ious) > 1 else 0.0,
        })
    return summary_rows


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Sweep --encoder-lr-scale across seeds for train_unet_finetune.py and summarize results."
    )
    parser.add_argument("--script-path", type=Path, default=Path("training/fine_tune.py"))
    parser.add_argument("--lr-scales", type=float, nargs="+", default=[0.1, 0.3, 0.5, 1.0, 2.0])
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    parser.add_argument("--sweep-results-dir", type=Path, default=Path("sweep_results"))
    parser.add_argument("--sweep-name", default="encoder_lr_sweep")

    # Forwarded as-is to train_unet_finetune.py; left as None means "let the
    # underlying script use its own default."
    parser.add_argument("--train-csv", type=Path, default=None)
    parser.add_argument("--val-csv", type=Path, default=None)
    parser.add_argument("--test-csv", type=Path, default=None)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--learning-rate", type=float, default=None)
    parser.add_argument("--weight-decay", type=float, default=None)
    parser.add_argument("--freeze-epochs", type=int, default=0)
    parser.add_argument("--base-channels", type=int, default=32)
    parser.add_argument("--pretrained-checkpoint", type=Path, default=None)
    parser.add_argument("--models-dir", type=Path, default=Path("models/sweep"))
    parser.add_argument("--results-dir", type=Path, default=Path("results/sweep"))
    parser.add_argument("--device", default=None)

    return parser.parse_args()


def main() -> None:
    args = parse_args()
    passthrough_args = build_passthrough_args(args)

    args.sweep_results_dir.mkdir(parents=True, exist_ok=True)
    per_run_csv = args.sweep_results_dir / f"{args.sweep_name}_per_run.csv"
    summary_csv = args.sweep_results_dir / f"{args.sweep_name}_summary.csv"

    total_runs = len(args.lr_scales) * len(args.seeds)
    print(f"Sweeping {len(args.lr_scales)} lr_scales x {len(args.seeds)} seeds = {total_runs} runs")

    all_results = []
    for lr_scale in args.lr_scales:
        for seed in args.seeds:
            run_name = f"{args.sweep_name}_lrscale{lr_scale}_seed{seed}".replace(".", "p")
            result = run_one(args.script_path, lr_scale, seed, run_name, passthrough_args)

            metrics_path = args.results_dir / f"{run_name}_metrics.csv"
            val_metrics = best_val_metrics_from_csv(metrics_path)
            if val_metrics is not None:
                result.update(val_metrics)

            all_results.append(result)

    with per_run_csv.open("w", newline="", encoding="utf-8") as handle:
        fieldnames = ["lr_scale", "seed", "run_name", "test_loss", "test_dice", "test_iou",
                      "val_loss", "val_dice", "val_iou", "best_epoch"]
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(all_results)

    summary_rows = summarize(all_results)
    with summary_csv.open("w", newline="", encoding="utf-8") as handle:
        fieldnames = ["lr_scale", "n_seeds", "test_dice_mean", "test_dice_std",
                      "test_iou_mean", "test_iou_std", "val_dice_mean", "val_dice_std",
                      "val_iou_mean", "val_iou_std"]
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(summary_rows)

    print("\n=== Sweep summary (mean +/- std across seeds) ===")
    for row in summary_rows:
        print(
            f"lr_scale={row['lr_scale']:<6} n_seeds={row['n_seeds']} "
            f"test_dice={row['test_dice_mean']:.4f} +/- {row['test_dice_std']:.4f}  "
            f"val_dice={row['val_dice_mean']:.4f} +/- {row['val_dice_std']:.4f}"
        )

    print(f"\nPer-run results: {per_run_csv}")
    print(f"Summary:         {summary_csv}")


if __name__ == "__main__":
    main()