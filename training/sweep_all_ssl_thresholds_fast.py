#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import importlib.util
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import DataLoader


METHODS = {
    "random": "lofo_{fp}_random_best.pt",
    "mae": "lofo_{fp}_ssl_best.pt",
    "contrastive": "lofo_{fp}_contrastive_best.pt",
    "denoising": "lofo_{fp}_denoising_best.pt",
    "weak_simsiam": "lofo_{fp}_weak_simsiam_best.pt",
    "hybrid": "lofo_{fp}_hybrid_best.pt",
}


def import_fine_tune(repo_root: Path) -> Any:
    path = repo_root / "training" / "fine_tune.py"
    spec = importlib.util.spec_from_file_location("fine_tune_module", path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not import {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@torch.no_grad()
def collect_probs_targets_loss(model, loader, loss_fn, device):
    model.eval()
    probs_list = []
    targets_list = []
    total_loss = 0.0
    batches = 0

    for images, masks in loader:
        images = images.to(device, non_blocking=True)
        masks = masks.to(device, non_blocking=True)

        logits = model(images)
        loss = loss_fn(logits, masks)

        probs_list.append(torch.sigmoid(logits).detach().flatten().cpu())
        targets_list.append((masks.detach().flatten().cpu() > 0.5))

        total_loss += float(loss.item())
        batches += 1

    if batches == 0:
        raise ValueError("No batches produced.")

    return torch.cat(probs_list), torch.cat(targets_list), total_loss / batches


def metrics_from_cached(probs, targets, loss, threshold):
    preds = probs > threshold
    targets = targets.bool()

    tp = int((preds & targets).sum().item())
    fp = int((preds & ~targets).sum().item())
    fn = int((~preds & targets).sum().item())
    tn = int((~preds & ~targets).sum().item())

    eps = 1e-7
    dice = (2 * tp + eps) / (2 * tp + fp + fn + eps)
    iou = (tp + eps) / (tp + fp + fn + eps)
    precision = (tp + eps) / (tp + fp + eps)
    recall = (tp + eps) / (tp + fn + eps)
    accuracy = (tp + tn + eps) / (tp + fp + fn + tn + eps)

    return {
        "loss": loss,
        "dice": dice,
        "iou": iou,
        "precision": precision,
        "recall": recall,
        "accuracy": accuracy,
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "tn": tn,
    }


def load_checkpoint(model, checkpoint_path, device):
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    state = checkpoint["model_state_dict"] if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint else checkpoint

    missing, unexpected = model.load_state_dict(state, strict=False)

    bad_missing = [key for key in missing if not key.startswith("out.")]
    if bad_missing:
        raise RuntimeError(f"{checkpoint_path} missing non-output keys: {bad_missing}")
    if unexpected:
        raise RuntimeError(f"{checkpoint_path} unexpected keys: {unexpected}")

    return checkpoint if isinstance(checkpoint, dict) else {}


def write_rows(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise ValueError(f"No rows for {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def sweep_one(
    fine_tune,
    repo_root,
    checkpoints_dir,
    results_dir,
    fp,
    method,
    thresholds,
    batch_size,
    num_workers,
    base_channels,
    device,
):
    split_dir = repo_root / "training" / "lofo_csvs" / f"heldout_{fp}"
    val_csv = split_dir / "validation.csv"
    test_csv = split_dir / "test.csv"
    checkpoint_path = checkpoints_dir / METHODS[method].format(fp=fp)

    for path in [val_csv, test_csv, checkpoint_path]:
        if not path.exists():
            raise FileNotFoundError(path)

    pin_memory = device.type == "cuda"

    val_loader = DataLoader(
        fine_tune.FloodTileDataset(val_csv),
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )
    test_loader = DataLoader(
        fine_tune.FloodTileDataset(test_csv),
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )

    model = fine_tune.UNet(in_channels=3, out_channels=1, base_channels=base_channels).to(device)
    checkpoint = load_checkpoint(model, checkpoint_path, device)
    loss_fn = fine_tune.FocalDiceLoss()

    print("  collecting validation predictions once")
    val_probs, val_targets, val_loss = collect_probs_targets_loss(model, val_loader, loss_fn, device)

    sweep_rows = []
    best = None
    for threshold in thresholds:
        m = metrics_from_cached(val_probs, val_targets, val_loss, threshold)
        row = {
            "heldout_fp": fp,
            "method": method,
            "threshold": threshold,
            "val_loss": float(m["loss"]),
            "val_dice": float(m["dice"]),
            "val_iou": float(m["iou"]),
            "val_precision": float(m["precision"]),
            "val_recall": float(m["recall"]),
            "val_accuracy": float(m["accuracy"]),
            "val_tp": int(m["tp"]),
            "val_fp": int(m["fp"]),
            "val_fn": int(m["fn"]),
            "val_tn": int(m["tn"]),
        }
        sweep_rows.append(row)

        if best is None or row["val_dice"] > best["val_dice"] or (
            row["val_dice"] == best["val_dice"] and row["val_iou"] > best["val_iou"]
        ):
            best = row

    assert best is not None
    selected_threshold = float(best["threshold"])

    print("  collecting test predictions once")
    test_probs, test_targets, test_loss = collect_probs_targets_loss(model, test_loader, loss_fn, device)
    test_m = metrics_from_cached(test_probs, test_targets, test_loss, selected_threshold)

    sweep_path = results_dir / f"lofo_{fp}_{method}_threshold_sweep.csv"
    write_rows(sweep_path, sweep_rows)

    return {
        "heldout_fp": fp,
        "method": method,
        "checkpoint_path": checkpoint_path.as_posix(),
        "selected_threshold": selected_threshold,
        "selected_val_dice": best["val_dice"],
        "selected_val_iou": best["val_iou"],
        "selected_val_precision": best["val_precision"],
        "selected_val_recall": best["val_recall"],
        "test_loss": float(test_m["loss"]),
        "test_dice": float(test_m["dice"]),
        "test_iou": float(test_m["iou"]),
        "test_precision": float(test_m["precision"]),
        "test_recall": float(test_m["recall"]),
        "test_accuracy": float(test_m["accuracy"]),
        "test_tp": int(test_m["tp"]),
        "test_fp": int(test_m["fp"]),
        "test_fn": int(test_m["fn"]),
        "test_tn": int(test_m["tn"]),
        "best_epoch": checkpoint.get("epoch", ""),
        "best_val_loss": checkpoint.get("best_val_loss", checkpoint.get("val_loss", "")),
        "sweep_csv": sweep_path.as_posix(),
    }


def build_summary(rows):
    by_key = {(row["heldout_fp"], row["method"]): row for row in rows}
    methods = ["random", "mae", "contrastive", "denoising", "weak_simsiam", "hybrid"]

    summary = []
    for i in range(1, 8):
        fp = f"fp{i}"
        row = {"heldout_fp": fp}

        for method in methods:
            r = by_key[(fp, method)]
            row[f"{method}_threshold"] = r["selected_threshold"]
            row[f"{method}_dice"] = r["test_dice"]
            row[f"{method}_iou"] = r["test_iou"]
            row[f"{method}_precision"] = r["test_precision"]
            row[f"{method}_recall"] = r["test_recall"]
            row[f"{method}_accuracy"] = r["test_accuracy"]
            row[f"{method}_best_epoch"] = r["best_epoch"]

        baseline_random = row["random_dice"]
        baseline_mae = row["mae_dice"]
        baseline_contrastive = row["contrastive_dice"]

        for method in ["denoising", "weak_simsiam", "hybrid"]:
            row[f"{method}_minus_random_dice"] = row[f"{method}_dice"] - baseline_random
            row[f"{method}_minus_mae_dice"] = row[f"{method}_dice"] - baseline_mae
            row[f"{method}_minus_contrastive_dice"] = row[f"{method}_dice"] - baseline_contrastive

        summary.append(row)

    return summary


def print_summary(summary):
    methods = ["random", "mae", "contrastive", "denoising", "weak_simsiam", "hybrid"]

    def mean(key):
        return sum(float(row[key]) for row in summary) / len(summary)

    print()
    print("===== All-method threshold-selected summary =====")
    for method in methods:
        print(
            f"{method}: "
            f"mean_dice={mean(f'{method}_dice'):.4f} "
            f"mean_iou={mean(f'{method}_iou'):.4f} "
            f"mean_precision={mean(f'{method}_precision'):.4f} "
            f"mean_recall={mean(f'{method}_recall'):.4f} "
            f"mean_accuracy={mean(f'{method}_accuracy'):.4f}"
        )

    print()
    for method in ["denoising", "weak_simsiam", "hybrid"]:
        wins_vs_random = sum(row[f"{method}_minus_random_dice"] > 0 for row in summary)
        wins_vs_mae = sum(row[f"{method}_minus_mae_dice"] > 0 for row in summary)
        wins_vs_contrastive = sum(row[f"{method}_minus_contrastive_dice"] > 0 for row in summary)
        print(
            f"{method}: "
            f"mean_minus_random={mean(f'{method}_minus_random_dice'):+.4f} "
            f"mean_minus_mae={mean(f'{method}_minus_mae_dice'):+.4f} "
            f"mean_minus_contrastive={mean(f'{method}_minus_contrastive_dice'):+.4f} "
            f"wins_vs_random={wins_vs_random}/7 "
            f"wins_vs_mae={wins_vs_mae}/7 "
            f"wins_vs_contrastive={wins_vs_contrastive}/7"
        )


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-root", type=Path, default=Path("."))
    parser.add_argument("--checkpoints-dir", type=Path, default=Path("models"))
    parser.add_argument("--results-dir", type=Path, default=Path("results"))
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--base-channels", type=int, default=32)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--threshold-start", type=float, default=0.05)
    parser.add_argument("--threshold-stop", type=float, default=0.95)
    parser.add_argument("--threshold-step", type=float, default=0.01)
    return parser.parse_args()


def main():
    args = parse_args()
    repo_root = args.repo_root.resolve()
    checkpoints_dir = (repo_root / args.checkpoints_dir).resolve()
    results_dir = (repo_root / args.results_dir).resolve()
    results_dir.mkdir(parents=True, exist_ok=True)

    fine_tune = import_fine_tune(repo_root)
    device = torch.device(args.device)

    thresholds = []
    t = args.threshold_start
    while t <= args.threshold_stop + 1e-9:
        thresholds.append(round(t, 4))
        t += args.threshold_step

    print(f"Using device: {device}")
    print(f"Methods: {list(METHODS.keys())}")
    print(f"Thresholds: {thresholds[0]} to {thresholds[-1]} step {args.threshold_step}")
    print("Fast mode: one validation pass and one test pass per checkpoint.")

    rows = []
    for i in range(1, 8):
        fp = f"fp{i}"
        for method in METHODS.keys():
            print(f"===== Sweeping {fp} {method} =====")
            row = sweep_one(
                fine_tune=fine_tune,
                repo_root=repo_root,
                checkpoints_dir=checkpoints_dir,
                results_dir=results_dir,
                fp=fp,
                method=method,
                thresholds=thresholds,
                batch_size=args.batch_size,
                num_workers=args.num_workers,
                base_channels=args.base_channels,
                device=device,
            )
            rows.append(row)
            print(
                f"{fp} {method}: "
                f"threshold={row['selected_threshold']:.2f} "
                f"val_dice={row['selected_val_dice']:.4f} "
                f"test_dice={row['test_dice']:.4f} "
                f"test_iou={row['test_iou']:.4f} "
                f"precision={row['test_precision']:.4f} "
                f"recall={row['test_recall']:.4f}"
            )

    all_runs_path = results_dir / "lofo_threshold_selected_all_6_methods.csv"
    write_rows(all_runs_path, rows)

    summary = build_summary(rows)
    summary_path = results_dir / "lofo_threshold_selected_summary_all_6_methods.csv"
    write_rows(summary_path, summary)

    print(f"Wrote {all_runs_path}")
    print(f"Wrote {summary_path}")
    print_summary(summary)


if __name__ == "__main__":
    main()