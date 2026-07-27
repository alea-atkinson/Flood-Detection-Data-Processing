#!/usr/bin/env python3

"""
Validation-threshold sweep for domain-balancing models.

For each trained domain-balancing model:
1. Load the best checkpoint.
2. Sweep thresholds on the validation set.
3. Select the threshold with best validation Dice.
4. Evaluate once on the held-out test set using that threshold.
5. Save run-level and summary results.

Inputs:
- domain_balancing/models/domain_balance_*_best.pt
- training/lofo_csvs/heldout_fp*/validation.csv
- training/lofo_csvs/heldout_fp*/test.csv

Outputs:
- domain_balancing/tables/domain_balancing_threshold_selected_all_runs.csv
- domain_balancing/tables/domain_balancing_threshold_selected_summary.csv
"""

from __future__ import annotations

import argparse
import csv
import re
import sys
from pathlib import Path
from statistics import mean, stdev

import torch
from torch.utils.data import DataLoader


REPO_ROOT = Path("/mnt/linuxlab/home/reuuzheng/Flood-Detection-Data-Processing")

DOMAIN_ROOT = REPO_ROOT / "domain_balancing"
MODELS_DIR = DOMAIN_ROOT / "models"
TABLES_DIR = DOMAIN_ROOT / "tables"
TABLES_DIR.mkdir(parents=True, exist_ok=True)

ALL_RUNS_OUT = TABLES_DIR / "domain_balancing_threshold_selected_all_runs.csv"
SUMMARY_OUT = TABLES_DIR / "domain_balancing_threshold_selected_summary.csv"

sys.path.insert(0, str(REPO_ROOT / "training"))

from fine_tune import FloodTileDataset, UNet  # noqa: E402


FOLDS = ["fp1", "fp2", "fp3", "fp4", "fp5", "fp6", "fp7"]

SAMPLING_STRATEGIES = [
    "standard",
    "source_fp",
    "flood_bin",
    "source_fp_x_flood_bin",
]

RUN_RE = re.compile(
    r"domain_balance_(standard|source_fp|flood_bin|source_fp_x_flood_bin)_(fp\d+)_seed(\d+)_best\.pt"
)

THRESHOLDS = [round(x / 100, 2) for x in range(5, 96)]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run validation-threshold selection for domain-balancing checkpoints."
    )
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument(
        "--device",
        default="cuda" if torch.cuda.is_available() else "cpu",
    )
    return parser.parse_args()


def split_csvs_for_fold(fold: str) -> dict[str, Path]:
    split_dir = REPO_ROOT / "training" / "lofo_csvs" / f"heldout_{fold}"
    return {
        "val": split_dir / "validation.csv",
        "test": split_dir / "test.csv",
    }


def get_state_dict(checkpoint):
    if not isinstance(checkpoint, dict):
        return checkpoint

    for key in [
        "model_state_dict",
        "state_dict",
        "model",
        "model_state",
        "net",
        "network",
    ]:
        if key in checkpoint and isinstance(checkpoint[key], dict):
            return checkpoint[key]

    tensor_like_values = [
        value for value in checkpoint.values()
        if hasattr(value, "shape")
    ]

    if tensor_like_values:
        return checkpoint

    raise KeyError(
        "Could not find model state dict in checkpoint. "
        f"Top-level keys are: {list(checkpoint.keys())}"
    )


def strip_module_prefix(state_dict: dict) -> dict:
    if not any(k.startswith("module.") for k in state_dict):
        return state_dict

    return {
        k.replace("module.", "", 1): v
        for k, v in state_dict.items()
    }


def infer_base_channels(state_dict: dict) -> int:
    preferred_keys = [
        "enc1.block.0.weight",
        "module.enc1.block.0.weight",
        "enc1.0.weight",
        "module.enc1.0.weight",
    ]

    for key in preferred_keys:
        if key in state_dict:
            return state_dict[key].shape[0]

    for key, value in state_dict.items():
        if (
            "enc1" in key
            and key.endswith(".weight")
            and hasattr(value, "shape")
            and len(value.shape) == 4
        ):
            print(f"Inferred base_channels from key: {key}")
            return value.shape[0]

    print("Could not infer base_channels. First 40 state_dict keys:")
    for i, key in enumerate(state_dict.keys()):
        if i >= 40:
            break
        print(" ", key)

    raise KeyError("Could not infer base_channels from checkpoint.")


def load_model(model_path: Path, device: torch.device) -> torch.nn.Module:
    checkpoint = torch.load(model_path, map_location=device, weights_only=False)
    state_dict = strip_module_prefix(get_state_dict(checkpoint))
    base_channels = infer_base_channels(state_dict)

    model = UNet(
        in_channels=3,
        out_channels=1,
        base_channels=base_channels,
    ).to(device)

    model.load_state_dict(state_dict, strict=True)
    model.eval()
    return model


def unpack_batch(batch):
    if isinstance(batch, dict):
        image_keys = ["image", "uavsar", "x", "inputs"]
        mask_keys = ["mask", "flood_mask", "y", "target", "label"]

        image = None
        mask = None

        for key in image_keys:
            if key in batch:
                image = batch[key]
                break

        for key in mask_keys:
            if key in batch:
                mask = batch[key]
                break

        if image is None or mask is None:
            raise KeyError(f"Could not unpack batch keys: {batch.keys()}")

        return image, mask

    if isinstance(batch, (list, tuple)):
        if len(batch) < 2:
            raise ValueError("Batch tuple/list has fewer than 2 items.")
        return batch[0], batch[1]

    raise TypeError(f"Unsupported batch type: {type(batch)}")


@torch.no_grad()
def collect_probs_and_masks(
    model: torch.nn.Module,
    csv_path: Path,
    device: torch.device,
    batch_size: int,
    num_workers: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    dataset = FloodTileDataset(csv_path)
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=(device.type == "cuda"),
    )

    all_probs = []
    all_masks = []

    for batch in loader:
        images, masks = unpack_batch(batch)

        images = images.to(device, non_blocking=True).float()
        masks = masks.to(device, non_blocking=True).float()

        logits = model(images)
        probs = torch.sigmoid(logits)

        if masks.ndim == 3:
            masks = masks.unsqueeze(1)

        all_probs.append(probs.detach().cpu().reshape(-1))
        all_masks.append(masks.detach().cpu().reshape(-1))

    probs = torch.cat(all_probs)
    masks = torch.cat(all_masks)
    masks = (masks > 0.5).float()

    return probs, masks


def metrics_from_probs(
    probs: torch.Tensor,
    masks: torch.Tensor,
    threshold: float,
) -> dict[str, float | int]:
    preds = probs >= threshold

    masks_bool = masks > 0.5

    tp = torch.sum(preds & masks_bool).item()
    fp = torch.sum(preds & ~masks_bool).item()
    fn = torch.sum(~preds & masks_bool).item()
    tn = torch.sum(~preds & ~masks_bool).item()

    eps = 1e-7

    dice = (2.0 * tp + eps) / (2.0 * tp + fp + fn + eps)
    iou = (tp + eps) / (tp + fp + fn + eps)
    precision = (tp + eps) / (tp + fp + eps)
    recall = (tp + eps) / (tp + fn + eps)
    accuracy = (tp + tn + eps) / (tp + fp + fn + tn + eps)

    return {
        "dice": dice,
        "iou": iou,
        "precision": precision,
        "recall": recall,
        "accuracy": accuracy,
        "tp": int(tp),
        "fp": int(fp),
        "fn": int(fn),
        "tn": int(tn),
    }


def select_threshold(
    val_probs: torch.Tensor,
    val_masks: torch.Tensor,
) -> tuple[float, dict[str, float | int]]:
    best_threshold = None
    best_metrics = None

    for threshold in THRESHOLDS:
        metrics = metrics_from_probs(val_probs, val_masks, threshold)

        if best_metrics is None or metrics["dice"] > best_metrics["dice"]:
            best_threshold = threshold
            best_metrics = metrics

    if best_threshold is None or best_metrics is None:
        raise RuntimeError("Threshold selection failed.")

    return best_threshold, best_metrics


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        raise ValueError(f"No rows to write for {path}")

    fieldnames = list(rows[0].keys())

    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def summarize(rows: list[dict]) -> list[dict]:
    metric_names = ["dice", "iou", "precision", "recall", "accuracy"]
    summary_rows = []

    groups = {}

    for row in rows:
        key = row["sampling_strategy"]
        groups.setdefault(key, []).append(row)

    for strategy in SAMPLING_STRATEGIES:
        group_rows = groups.get(strategy, [])

        if not group_rows:
            continue

        out = {
            "sampling_strategy": strategy,
            "n": len(group_rows),
        }

        for metric in metric_names:
            values = [float(r[f"test_{metric}"]) for r in group_rows]
            out[f"mean_{metric}"] = mean(values)
            out[f"std_{metric}"] = stdev(values) if len(values) > 1 else 0.0
            out[f"min_{metric}"] = min(values)
            out[f"max_{metric}"] = max(values)

        threshold_values = [float(r["selected_threshold"]) for r in group_rows]
        out["mean_selected_threshold"] = mean(threshold_values)
        out["std_selected_threshold"] = (
            stdev(threshold_values) if len(threshold_values) > 1 else 0.0
        )
        out["min_selected_threshold"] = min(threshold_values)
        out["max_selected_threshold"] = max(threshold_values)

        summary_rows.append(out)

    summary_rows = sorted(summary_rows, key=lambda r: r["mean_dice"], reverse=True)
    return summary_rows


def main() -> None:
    args = parse_args()
    device = torch.device(args.device)

    print(f"Using device: {device}")

    model_files = sorted(MODELS_DIR.glob("domain_balance_*_best.pt"))

    # Exclude old mistaken smoke-style/manual files if any happen to match loosely.
    model_files = [
        p for p in model_files
        if RUN_RE.match(p.name)
    ]

    if len(model_files) != 28:
        print(f"WARNING: expected 28 model files, found {len(model_files)}")

    rows = []

    for model_path in model_files:
        match = RUN_RE.match(model_path.name)

        if not match:
            print(f"Skipping unexpected model file: {model_path.name}")
            continue

        strategy, fold, seed = match.groups()
        seed = int(seed)

        print("\n" + "=" * 80)
        print(f"Evaluating {model_path.name}")
        print(f"Fold: {fold} | Strategy: {strategy} | Seed: {seed}")
        print("=" * 80)

        csvs = split_csvs_for_fold(fold)

        model = load_model(model_path, device)

        val_probs, val_masks = collect_probs_and_masks(
            model=model,
            csv_path=csvs["val"],
            device=device,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
        )

        selected_threshold, val_metrics = select_threshold(val_probs, val_masks)

        test_probs, test_masks = collect_probs_and_masks(
            model=model,
            csv_path=csvs["test"],
            device=device,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
        )

        test_metrics = metrics_from_probs(
            test_probs,
            test_masks,
            selected_threshold,
        )

        row = {
            "run_name": f"domain_balance_{strategy}_{fold}_seed{seed}",
            "fold": fold,
            "sampling_strategy": strategy,
            "seed": seed,
            "model_path": str(model_path),
            "val_csv": str(csvs["val"]),
            "test_csv": str(csvs["test"]),
            "selected_threshold": selected_threshold,

            "val_dice": val_metrics["dice"],
            "val_iou": val_metrics["iou"],
            "val_precision": val_metrics["precision"],
            "val_recall": val_metrics["recall"],
            "val_accuracy": val_metrics["accuracy"],

            "test_dice": test_metrics["dice"],
            "test_iou": test_metrics["iou"],
            "test_precision": test_metrics["precision"],
            "test_recall": test_metrics["recall"],
            "test_accuracy": test_metrics["accuracy"],
            "test_tp": test_metrics["tp"],
            "test_fp": test_metrics["fp"],
            "test_fn": test_metrics["fn"],
            "test_tn": test_metrics["tn"],
        }

        print(
            f"Selected threshold: {selected_threshold:.2f} | "
            f"val Dice: {val_metrics['dice']:.4f} | "
            f"test Dice: {test_metrics['dice']:.4f}"
        )

        rows.append(row)

        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    rows = sorted(rows, key=lambda r: (r["fold"], r["sampling_strategy"], r["seed"]))

    write_csv(ALL_RUNS_OUT, rows)

    summary_rows = summarize(rows)
    write_csv(SUMMARY_OUT, summary_rows)

    print()
    print(f"Wrote {ALL_RUNS_OUT}")
    print(f"Wrote {SUMMARY_OUT}")
    print()
    print("Overall validation-threshold-selected domain-balancing summary:")
    for row in summary_rows:
        print(
            f"{row['sampling_strategy']:25s} "
            f"Dice {row['mean_dice']:.4f} ± {row['std_dice']:.4f} | "
            f"IoU {row['mean_iou']:.4f} ± {row['std_iou']:.4f} | "
            f"Precision {row['mean_precision']:.4f} | "
            f"Recall {row['mean_recall']:.4f} | "
            f"Threshold {row['mean_selected_threshold']:.3f}"
        )


if __name__ == "__main__":
    main()