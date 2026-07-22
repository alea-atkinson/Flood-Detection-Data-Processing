#!/usr/bin/env python3

"""
Validation-threshold sweep for seed replication models.

For each trained seed-replication model:
1. Load the best checkpoint.
2. Sweep thresholds on the validation set.
3. Select the threshold with best validation Dice.
4. Evaluate once on the held-out test set using that threshold.
5. Save all run-level and summary results.

Inputs:
- seed_replication/models/seedrep_*_best.pt
- training/lofo_csvs/heldout_fp*/train.csv
- training/lofo_csvs/heldout_fp*/validation.csv
- training/lofo_csvs/heldout_fp*/test.csv

Outputs:
- seed_replication/tables/seed_replication_threshold_selected_all_runs.csv
- seed_replication/tables/seed_replication_threshold_selected_summary.csv
"""

import csv
import re
import sys
from pathlib import Path
from statistics import mean, stdev

import torch
from torch.utils.data import DataLoader


REPO_ROOT = Path("/mnt/linuxlab/home/reuuzheng/Flood-Detection-Data-Processing")
SEED_ROOT = REPO_ROOT / "seed_replication"
MODELS_DIR = SEED_ROOT / "models"
TABLES_DIR = SEED_ROOT / "tables"
TABLES_DIR.mkdir(parents=True, exist_ok=True)

ALL_RUNS_OUT = TABLES_DIR / "seed_replication_threshold_selected_all_runs.csv"
SUMMARY_OUT = TABLES_DIR / "seed_replication_threshold_selected_summary.csv"

sys.path.insert(0, str(REPO_ROOT / "training"))

from fine_tune import FloodTileDataset, UNet  # noqa: E402


FOLDS = ["fp1", "fp2", "fp3", "fp4", "fp5", "fp6", "fp7"]
METHODS = ["random", "weak_simsiam"]
SEEDS = [1, 2, 3]

RUN_RE = re.compile(r"seedrep_(fp\d+)_(random|weak_simsiam)_seed(\d+)_best\.pt")

THRESHOLDS = [round(x / 100, 2) for x in range(5, 96)]


def split_csvs_for_fold(fold: str) -> dict[str, Path]:
    split_dir = REPO_ROOT / "training" / "lofo_csvs" / f"heldout_{fold}"

    return {
        "train": split_dir / "train.csv",
        "val": split_dir / "validation.csv",
        "test": split_dir / "test.csv",
    }


def get_state_dict(checkpoint):
    if not isinstance(checkpoint, dict):
        return checkpoint

    # Common checkpoint formats.
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

    # Sometimes the checkpoint itself is already a state_dict.
    # A real state_dict usually has tensor values.
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


def infer_base_channels(state_dict: dict) -> int:
    """
    Infer U-Net base channels from the first convolution weight.
    Handles several naming conventions.
    """

    preferred_keys = [
        "enc1.0.weight",
        "encoder.enc1.0.weight",
        "unet.enc1.0.weight",
        "model.enc1.0.weight",
        "module.enc1.0.weight",
    ]

    for key in preferred_keys:
        if key in state_dict:
            return state_dict[key].shape[0]

    # Flexible fallback: find the first enc1 conv weight.
    for key, value in state_dict.items():
        if (
            key.endswith("enc1.0.weight")
            or "enc1.0.weight" in key
            or key.endswith("enc1.conv.0.weight")
            or "enc1" in key and key.endswith(".weight") and len(value.shape) == 4
        ):
            print(f"Inferred base_channels from key: {key}")
            return value.shape[0]

    print("Could not infer base_channels. First 40 state_dict keys:")
    for i, key in enumerate(state_dict.keys()):
        if i >= 40:
            break
        print(" ", key)

    raise KeyError("Could not infer base_channels from checkpoint.")


def strip_module_prefix(state_dict: dict) -> dict:
    if not any(k.startswith("module.") for k in state_dict):
        return state_dict

    return {
        k.replace("module.", "", 1): v
        for k, v in state_dict.items()
    }


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
    """
    Supports common dataset return formats:
    - (image, mask)
    - (image, mask, ...)
    - {"image": ..., "mask": ...}
    - {"uavsar": ..., "flood_mask": ...}
    """
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
    batch_size: int = 8,
) -> tuple[torch.Tensor, torch.Tensor]:
    dataset = FloodTileDataset(csv_path)
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=torch.cuda.is_available(),
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
) -> dict[str, float]:
    preds = (probs >= threshold).float()

    tp = torch.sum((preds == 1) & (masks == 1)).item()
    fp = torch.sum((preds == 1) & (masks == 0)).item()
    fn = torch.sum((preds == 0) & (masks == 1)).item()
    tn = torch.sum((preds == 0) & (masks == 0)).item()

    eps = 1e-7

    dice = (2 * tp) / (2 * tp + fp + fn + eps)
    iou = tp / (tp + fp + fn + eps)
    precision = tp / (tp + fp + eps)
    recall = tp / (tp + fn + eps)
    accuracy = (tp + tn) / (tp + fp + fn + tn + eps)

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
) -> tuple[float, dict[str, float]]:
    best_threshold = None
    best_metrics = None

    for threshold in THRESHOLDS:
        metrics = metrics_from_probs(val_probs, val_masks, threshold)

        if best_metrics is None or metrics["dice"] > best_metrics["dice"]:
            best_threshold = threshold
            best_metrics = metrics

    return best_threshold, best_metrics


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        raise ValueError(f"No rows to write for {path}")

    fieldnames = list(rows[0].keys())

    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def summarize(rows: list[dict]) -> list[dict]:
    metric_names = ["dice", "iou", "precision", "recall", "accuracy"]
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

        for metric in metric_names:
            values = [float(r[f"test_{metric}"]) for r in group_rows]
            out[f"{metric}_mean"] = mean(values)
            out[f"{metric}_std"] = stdev(values) if len(values) > 1 else 0.0
            out[f"{metric}_min"] = min(values)
            out[f"{metric}_max"] = max(values)

        threshold_values = [float(r["selected_threshold"]) for r in group_rows]
        out["threshold_mean"] = mean(threshold_values)
        out["threshold_std"] = stdev(threshold_values) if len(threshold_values) > 1 else 0.0
        out["threshold_min"] = min(threshold_values)
        out["threshold_max"] = max(threshold_values)

        summary_rows.append(out)

    for method in METHODS:
        method_rows = [r for r in rows if r["method"] == method]

        out = {
            "fold": "ALL",
            "method": method,
            "n_seeds": len(method_rows),
        }

        for metric in metric_names:
            values = [float(r[f"test_{metric}"]) for r in method_rows]
            out[f"{metric}_mean"] = mean(values)
            out[f"{metric}_std"] = stdev(values) if len(values) > 1 else 0.0
            out[f"{metric}_min"] = min(values)
            out[f"{metric}_max"] = max(values)

        threshold_values = [float(r["selected_threshold"]) for r in method_rows]
        out["threshold_mean"] = mean(threshold_values)
        out["threshold_std"] = stdev(threshold_values) if len(threshold_values) > 1 else 0.0
        out["threshold_min"] = min(threshold_values)
        out["threshold_max"] = max(threshold_values)

        summary_rows.append(out)

    return summary_rows


def main() -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    model_files = sorted(MODELS_DIR.glob("seedrep_*_best.pt"))

    if len(model_files) != 42:
        print(f"WARNING: expected 42 model files, found {len(model_files)}")

    rows = []

    for model_path in model_files:
        match = RUN_RE.match(model_path.name)

        if not match:
            print(f"Skipping unexpected model file: {model_path.name}")
            continue

        fold, method, seed = match.groups()
        seed = int(seed)

        print("\n" + "=" * 80)
        print(f"Evaluating {model_path.name}")
        print(f"Fold: {fold} | Method: {method} | Seed: {seed}")
        print("=" * 80)

        csvs = split_csvs_for_fold(fold)

        model = load_model(model_path, device)

        val_probs, val_masks = collect_probs_and_masks(
            model=model,
            csv_path=csvs["val"],
            device=device,
        )

        selected_threshold, val_metrics = select_threshold(val_probs, val_masks)

        test_probs, test_masks = collect_probs_and_masks(
            model=model,
            csv_path=csvs["test"],
            device=device,
        )

        test_metrics = metrics_from_probs(
            test_probs,
            test_masks,
            selected_threshold,
        )

        row = {
            "run_name": f"seedrep_{fold}_{method}_seed{seed}",
            "fold": fold,
            "method": method,
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

    rows = sorted(rows, key=lambda r: (r["fold"], r["method"], r["seed"]))

    write_csv(ALL_RUNS_OUT, rows)

    summary_rows = summarize(rows)
    write_csv(SUMMARY_OUT, summary_rows)

    print(f"\nWrote {ALL_RUNS_OUT}")
    print(f"Wrote {SUMMARY_OUT}")

    random_all = next(
        r for r in summary_rows
        if r["fold"] == "ALL" and r["method"] == "random"
    )
    weak_all = next(
        r for r in summary_rows
        if r["fold"] == "ALL" and r["method"] == "weak_simsiam"
    )

    print("\nOverall validation-threshold-selected seed replication:")
    print(
        f"Random Dice mean ± std:       "
        f"{random_all['dice_mean']:.4f} ± {random_all['dice_std']:.4f}"
    )
    print(
        f"Weak SimSiam Dice mean ± std: "
        f"{weak_all['dice_mean']:.4f} ± {weak_all['dice_std']:.4f}"
    )
    print(
        f"Difference:                   "
        f"{weak_all['dice_mean'] - random_all['dice_mean']:+.4f}"
    )

    print("\nOverall IoU:")
    print(
        f"Random IoU mean ± std:        "
        f"{random_all['iou_mean']:.4f} ± {random_all['iou_std']:.4f}"
    )
    print(
        f"Weak SimSiam IoU mean ± std:  "
        f"{weak_all['iou_mean']:.4f} ± {weak_all['iou_std']:.4f}"
    )
    print(
        f"Difference:                   "
        f"{weak_all['iou_mean'] - random_all['iou_mean']:+.4f}"
    )


if __name__ == "__main__":
    main()