#!/usr/bin/env python3
"""Fine-tune a SAR-only U-Net for flood/change segmentation.

Supports either:
1. random initialization, or
2. optional initialization from a pretrained checkpoint.

Example random-init run:
    python3 training/milton_pretraining.py \
        --train-csv training/fine_tune_csvs/no_overlap_converted/held_out_fp1_train.csv \
        --val-csv training/fine_tune_csvs/no_overlap_converted/held_out_fp1_val.csv \
        --test-csv training/fine_tune_csvs/no_overlap_converted/held_out_fp1_test.csv \
        --epochs 20 \
        --batch-size 16 \
        --num-workers 0 \
        --models-dir drought_augmentation/models \
        --results-dir drought_augmentation/raw_results \
        --run-name fp1_random_init

Example pretrained run:
    python3 training/milton_pretraining.py \
        --pretrained_weights drought_augmentation/models/drought_negative_pretraining_best.pt \
        --train-csv training/fine_tune_csvs/no_overlap_converted/held_out_fp1_train.csv \
        --val-csv training/fine_tune_csvs/no_overlap_converted/held_out_fp1_val.csv \
        --test-csv training/fine_tune_csvs/no_overlap_converted/held_out_fp1_test.csv \
        --epochs 20 \
        --batch-size 16 \
        --num-workers 0 \
        --models-dir drought_augmentation/models \
        --results-dir drought_augmentation/raw_results \
        --run-name fp1_drought_negative_init
"""

from __future__ import annotations

import argparse
import csv
import importlib.util
import random
import sys
from pathlib import Path

import numpy as np


def print_basic_help() -> None:
    """Show help even before optional training packages are installed."""
    print(
        """usage: milton_pretraining.py [options]

Train/fine-tune a SAR-only binary U-Net.

Required:
  --train-csv PATH        Training split CSV
  --val-csv PATH          Validation split CSV
  --test-csv PATH         Test split CSV

Common options:
  --pretrained_weights PATH Optional checkpoint path. If omitted, train from random initialization.
  --epochs N               Number of epochs
  --batch-size N           Batch size
  --learning-rate LR       AdamW learning rate
  --weight-decay WD        AdamW weight decay
  --num-workers N          DataLoader workers
  --base-channels N        U-Net width
  --seed N                 Random seed
  --models-dir PATH        Checkpoint output folder
  --results-dir PATH       Metrics CSV output folder
  --run-name NAME          Prefix for output files
  --device DEVICE          cuda or cpu
"""
    )


if any(arg in {"-h", "--help"} for arg in sys.argv[1:]):
    print_basic_help()
    raise SystemExit(0)


def require_package(package_name: str) -> None:
    if importlib.util.find_spec(package_name) is None:
        raise SystemExit(
            f"Missing required package: {package_name}\n"
            f"Install the project environment packages, then rerun this script. "
            f"For example, check with: python3 -c \"import {package_name}\""
        )


require_package("torch")
require_package("rasterio")

import rasterio  # noqa: E402
import torch  # noqa: E402
import torch.nn as nn  # noqa: E402
import torch.nn.functional as F  # noqa: E402
from torch.utils.data import DataLoader, Dataset  # noqa: E402


class FloodTileDataset(Dataset):
    """Reads SAR image tiles and binary flood/change masks from split CSV rows."""

    def __init__(self, csv_path: Path) -> None:
        self.csv_path = csv_path
        self.rows = self._read_rows(csv_path)

    @staticmethod
    def _read_rows(csv_path: Path) -> list[dict[str, str]]:
        with csv_path.open("r", newline="", encoding="utf-8") as handle:
            reader = csv.DictReader(handle)
            required_columns = {"uavsar_path", "flood_mask_path"}
            missing = required_columns - set(reader.fieldnames or [])
            if missing:
                raise ValueError(f"{csv_path} is missing columns: {sorted(missing)}")
            return list(reader)

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int):
        row = self.rows[index]

        sar_path = row["uavsar_path"]
        mask_path = row["flood_mask_path"]

        with rasterio.open(sar_path) as src:
            sar = src.read().astype(np.float32)

        if sar.shape[0] < 3:
            raise ValueError(
                f"Expected at least 3 SAR bands, got {sar.shape[0]} in {sar_path}"
            )

        sar = sar[:3]

        # Valid SAR pixels: valid unless all first three bands are zero.
        sar_valid = ~(sar == 0).all(axis=0)
        sar[:, ~sar_valid] = np.nan

        with rasterio.open(mask_path) as src:
            mask = src.read(1)

        # Treat 255 / values >= 200 as nodata.
        valid_mask = mask < 200

        # Current verified masks use 1 = flood, 0 = non-flood, 255 = nodata.
        binary_mask = np.zeros(mask.shape, dtype=np.float32)
        binary_mask[(mask == 1) & valid_mask] = 1.0

        sar = self._normalize_per_tile(sar, sar_valid)

        valid = sar_valid & valid_mask

        return (
            torch.from_numpy(sar),
            torch.from_numpy(binary_mask[None]),
            torch.from_numpy(valid.astype(np.float32)[None]),
        )

    @staticmethod
    def _normalize_per_tile(sar: np.ndarray, valid: np.ndarray) -> np.ndarray:
        sar = sar.copy()

        for c in range(sar.shape[0]):
            band = sar[c]
            values = band[valid]

            if values.size == 0:
                band[:] = 0.0
                sar[c] = band
                continue

            low, high = np.percentile(values, [1.0, 99.0])
            clipped_values = np.clip(values, low, high)

            mean = clipped_values.mean()
            std = clipped_values.std()

            if std < 1e-6:
                band[:] = 0.0
            else:
                band[valid] = (clipped_values - mean) / std

            band[~valid] = 0.0
            sar[c] = band

        return sar.astype(np.float32)


class DoubleConv(nn.Module):
    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class UNet(nn.Module):
    """Small plain U-Net for binary segmentation."""

    def __init__(
        self,
        in_channels: int = 3,
        out_channels: int = 1,
        base_channels: int = 32,
    ) -> None:
        super().__init__()

        self.enc1 = DoubleConv(in_channels, base_channels)
        self.enc2 = DoubleConv(base_channels, base_channels * 2)
        self.enc3 = DoubleConv(base_channels * 2, base_channels * 4)
        self.enc4 = DoubleConv(base_channels * 4, base_channels * 8)

        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.bottleneck = DoubleConv(base_channels * 8, base_channels * 16)

        self.up4 = nn.ConvTranspose2d(
            base_channels * 16,
            base_channels * 8,
            kernel_size=2,
            stride=2,
        )
        self.dec4 = DoubleConv(base_channels * 16, base_channels * 8)

        self.up3 = nn.ConvTranspose2d(
            base_channels * 8,
            base_channels * 4,
            kernel_size=2,
            stride=2,
        )
        self.dec3 = DoubleConv(base_channels * 8, base_channels * 4)

        self.up2 = nn.ConvTranspose2d(
            base_channels * 4,
            base_channels * 2,
            kernel_size=2,
            stride=2,
        )
        self.dec2 = DoubleConv(base_channels * 4, base_channels * 2)

        self.up1 = nn.ConvTranspose2d(
            base_channels * 2,
            base_channels,
            kernel_size=2,
            stride=2,
        )
        self.dec1 = DoubleConv(base_channels * 2, base_channels)

        self.out = nn.Conv2d(base_channels, out_channels, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        enc1 = self.enc1(x)
        enc2 = self.enc2(self.pool(enc1))
        enc3 = self.enc3(self.pool(enc2))
        enc4 = self.enc4(self.pool(enc3))

        x = self.bottleneck(self.pool(enc4))

        x = self.up4(x)
        x = self.dec4(torch.cat([x, enc4], dim=1))

        x = self.up3(x)
        x = self.dec3(torch.cat([x, enc3], dim=1))

        x = self.up2(x)
        x = self.dec2(torch.cat([x, enc2], dim=1))

        x = self.up1(x)
        x = self.dec1(torch.cat([x, enc1], dim=1))

        return self.out(x)


def segmentation_metrics_from_logits(
    logits: torch.Tensor,
    targets: torch.Tensor,
    mask: torch.Tensor,
    threshold: float = 0.5,
) -> dict[str, torch.Tensor]:
    probs = torch.sigmoid(logits)
    preds = probs > threshold
    targets_bool = targets > 0.5

    valid = mask > 0.5

    preds = preds & valid
    targets_bool = targets_bool & valid

    true_positive = (preds & targets_bool).sum().float()
    false_positive = (preds & ~targets_bool & valid).sum().float()
    false_negative = (~preds & targets_bool & valid).sum().float()

    pred_sum = preds.sum().float()
    target_sum = targets_bool.sum().float()

    return {
        "tp": true_positive,
        "fp": false_positive,
        "fn": false_negative,
        "pred_sum": pred_sum,
        "target_sum": target_sum,
    }


class DiceLoss(nn.Module):
    def __init__(self, smooth: float = 1.0) -> None:
        super().__init__()
        self.smooth = smooth

    def forward(
        self,
        logits: torch.Tensor,
        targets: torch.Tensor,
        mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        probs = torch.sigmoid(logits)

        if mask is not None:
            probs = probs * mask
            targets = targets * mask

        probs = probs.view(-1)
        targets = targets.view(-1)

        intersection = (probs * targets).sum()
        denom = probs.sum() + targets.sum()

        dice = (2.0 * intersection + self.smooth) / (denom + self.smooth)
        return 1 - dice


class FocalLoss(nn.Module):
    """Masked focal BCE loss."""

    def __init__(self, alpha: float = 0.25, gamma: float = 2.0) -> None:
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(
        self,
        logits: torch.Tensor,
        targets: torch.Tensor,
        mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        bce = F.binary_cross_entropy_with_logits(
            logits,
            targets,
            reduction="none",
        )

        probs = torch.sigmoid(logits)
        pt = torch.where(targets == 1, probs, 1 - probs)

        focal_weight = self.alpha * (1 - pt) ** self.gamma
        loss = focal_weight * bce

        if mask is not None:
            loss = loss * mask
            return loss.sum() / (mask.sum() + 1e-6)

        return loss.mean()


class FocalDiceLoss(nn.Module):
    """Focal + Dice loss for class-imbalanced segmentation."""

    def __init__(
        self,
        alpha: float = 0.25,
        gamma: float = 2.0,
        dice_weight: float = 1.0,
        focal_weight: float = 1.0,
    ) -> None:
        super().__init__()
        self.focal = FocalLoss(alpha, gamma)
        self.dice = DiceLoss()
        self.dice_weight = dice_weight
        self.focal_weight = focal_weight

    def forward(
        self,
        logits: torch.Tensor,
        targets: torch.Tensor,
        mask: torch.Tensor,
    ) -> torch.Tensor:
        focal_loss = self.focal(logits, targets, mask)
        dice_loss = self.dice(logits, targets, mask)
        return self.focal_weight * focal_loss + self.dice_weight * dice_loss


def run_epoch(
    model: nn.Module,
    loader: DataLoader,
    loss_fn: nn.Module,
    device: torch.device,
    optimizer: torch.optim.Optimizer | None = None,
) -> dict[str, float]:
    is_train = optimizer is not None
    model.train(is_train)

    total_loss = 0.0
    total_tp = 0.0
    total_fp = 0.0
    total_fn = 0.0
    total_pred = 0.0
    total_target = 0.0
    total_batches = 0

    for images, masks, valid in loader:
        images = images.to(device, non_blocking=True)
        masks = masks.to(device, non_blocking=True)
        valid = valid.to(device, non_blocking=True)

        with torch.set_grad_enabled(is_train):
            logits = model(images)
            loss = loss_fn(logits, masks, valid)

            if is_train:
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step()

        metrics = segmentation_metrics_from_logits(
            logits.detach(),
            masks,
            valid,
        )

        total_loss += float(loss.item())

        total_tp += metrics["tp"].item()
        total_fp += metrics["fp"].item()
        total_fn += metrics["fn"].item()
        total_pred += metrics["pred_sum"].item()
        total_target += metrics["target_sum"].item()

        total_batches += 1

    if total_batches == 0:
        raise ValueError("DataLoader produced no batches.")

    eps = 1e-7

    global_dice = (2 * total_tp + eps) / (total_pred + total_target + eps)
    global_iou = (total_tp + eps) / (total_pred + total_target - total_tp + eps)
    global_precision = (total_tp + eps) / (total_tp + total_fp + eps)
    global_recall = (total_tp + eps) / (total_tp + total_fn + eps)

    return {
        "loss": total_loss / total_batches,
        "dice": global_dice,
        "iou": global_iou,
        "precision": global_precision,
        "recall": global_recall,
    }


def write_metrics_csv(metrics_path: Path, rows: list[dict[str, float | int]]) -> None:
    metrics_path.parent.mkdir(parents=True, exist_ok=True)

    with metrics_path.open("w", newline="", encoding="utf-8") as handle:
        fieldnames = [
            "epoch",
            "train_loss",
            "train_dice",
            "train_iou",
            "train_precision",
            "train_recall",
            "val_loss",
            "val_dice",
            "val_iou",
            "val_precision",
            "val_recall",
        ]
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_test_metrics_csv(
    test_metrics_path: Path,
    args: argparse.Namespace,
    checkpoint: dict,
    test_metrics: dict[str, float],
) -> None:
    test_metrics_path.parent.mkdir(parents=True, exist_ok=True)

    with test_metrics_path.open("w", newline="", encoding="utf-8") as handle:
        fieldnames = [
            "run_name",
            "train_csv",
            "val_csv",
            "test_csv",
            "pretrained_weights",
            "best_epoch",
            "best_val_loss",
            "test_loss",
            "test_dice",
            "test_iou",
            "test_precision",
            "test_recall",
        ]

        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerow(
            {
                "run_name": args.run_name,
                "train_csv": str(args.train_csv),
                "val_csv": str(args.val_csv),
                "test_csv": str(args.test_csv),
                "pretrained_weights": str(args.pretrained_weights)
                if args.pretrained_weights
                else "",
                "best_epoch": checkpoint.get("epoch", ""),
                "best_val_loss": checkpoint.get("best_val_loss", ""),
                "test_loss": test_metrics["loss"],
                "test_dice": test_metrics["dice"],
                "test_iou": test_metrics["iou"],
                "test_precision": test_metrics["precision"],
                "test_recall": test_metrics["recall"],
            }
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Fine-tune a SAR-only binary U-Net with optional pretrained weights."
    )

    parser.add_argument("--train-csv", type=Path, required=True)
    parser.add_argument("--val-csv", type=Path, required=True)
    parser.add_argument("--test-csv", type=Path, required=True)

    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--learning-rate", type=float, default=9.327106954111342e-05)
    parser.add_argument("--weight-decay", type=float, default=6.088353841746043e-06)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--base-channels", type=int, default=32)
    parser.add_argument("--seed", type=int, default=42)

    parser.add_argument("--models-dir", type=Path, default=Path("models"))
    parser.add_argument("--results-dir", type=Path, default=Path("results"))
    parser.add_argument("--run-name", required=True)

    parser.add_argument(
        "--pretrained_weights",
        "--pretrained-weights",
        dest="pretrained_weights",
        type=Path,
        default=None,
        help=(
            "Optional checkpoint path for initializing the model. "
            "If omitted, train from random initialization."
        ),
    )

    parser.add_argument(
        "--device",
        default="cuda" if torch.cuda.is_available() else "cpu",
    )

    return parser.parse_args()


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def main() -> None:
    args = parse_args()
    set_seed(args.seed)

    device = torch.device(args.device)

    print(f"Using device: {device}")
    print(f"Train CSV: {args.train_csv}")
    print(f"Validation CSV: {args.val_csv}")
    print(f"Test CSV: {args.test_csv}")

    train_dataset = FloodTileDataset(args.train_csv)
    val_dataset = FloodTileDataset(args.val_csv)
    test_dataset = FloodTileDataset(args.test_csv)

    print(
        f"Dataset sizes: "
        f"train={len(train_dataset)}, "
        f"val={len(val_dataset)}, "
        f"test={len(test_dataset)}"
    )

    pin_memory = device.type == "cuda"

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=pin_memory,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=pin_memory,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=pin_memory,
    )

    model = UNet(
        in_channels=3,
        out_channels=1,
        base_channels=args.base_channels,
    ).to(device)

    if args.pretrained_weights:
        print(f"Loading pretrained weights from: {args.pretrained_weights}")

        checkpoint = torch.load(
            args.pretrained_weights,
            map_location=device,
            weights_only=False,
        )

        if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
            model.load_state_dict(checkpoint["model_state_dict"])
        else:
            model.load_state_dict(checkpoint)
    else:
        print("No pretrained weights provided. Training from random initialization.")

    loss_fn = FocalDiceLoss()

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
    )

    args.models_dir.mkdir(parents=True, exist_ok=True)
    args.results_dir.mkdir(parents=True, exist_ok=True)

    checkpoint_path = args.models_dir / f"{args.run_name}_best.pt"
    metrics_path = args.results_dir / f"{args.run_name}_metrics.csv"
    test_metrics_path = args.results_dir / f"{args.run_name}_test_metrics.csv"

    best_val_loss = float("inf")
    history: list[dict[str, float | int]] = []

    for epoch in range(1, args.epochs + 1):
        train_metrics = run_epoch(model, train_loader, loss_fn, device, optimizer)
        val_metrics = run_epoch(model, val_loader, loss_fn, device)

        row = {
            "epoch": epoch,
            "train_loss": train_metrics["loss"],
            "train_dice": train_metrics["dice"],
            "train_iou": train_metrics["iou"],
            "train_precision": train_metrics["precision"],
            "train_recall": train_metrics["recall"],
            "val_loss": val_metrics["loss"],
            "val_dice": val_metrics["dice"],
            "val_iou": val_metrics["iou"],
            "val_precision": val_metrics["precision"],
            "val_recall": val_metrics["recall"],
        }

        history.append(row)
        write_metrics_csv(metrics_path, history)

        print(
            f"Epoch {epoch:03d}/{args.epochs:03d} "
            f"train_loss={train_metrics['loss']:.4f} "
            f"train_dice={train_metrics['dice']:.4f} "
            f"train_iou={train_metrics['iou']:.4f} "
            f"train_precision={train_metrics['precision']:.4f} "
            f"train_recall={train_metrics['recall']:.4f} "
            f"val_loss={val_metrics['loss']:.4f} "
            f"val_dice={val_metrics['dice']:.4f} "
            f"val_iou={val_metrics['iou']:.4f} "
            f"val_precision={val_metrics['precision']:.4f} "
            f"val_recall={val_metrics['recall']:.4f}"
        )

        if val_metrics["loss"] < best_val_loss:
            best_val_loss = val_metrics["loss"]

            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "best_val_loss": best_val_loss,
                    "args": vars(args),
                },
                checkpoint_path,
            )

            print(f"  Saved best checkpoint: {checkpoint_path}")

    checkpoint = torch.load(
        checkpoint_path,
        map_location=device,
        weights_only=False,
    )

    model.load_state_dict(checkpoint["model_state_dict"])

    test_metrics = run_epoch(model, test_loader, loss_fn, device)

    print(
        "Best checkpoint test metrics: "
        f"loss={test_metrics['loss']:.4f} "
        f"dice={test_metrics['dice']:.4f} "
        f"iou={test_metrics['iou']:.4f} "
        f"precision={test_metrics['precision']:.4f} "
        f"recall={test_metrics['recall']:.4f}"
    )

    write_test_metrics_csv(
        test_metrics_path=test_metrics_path,
        args=args,
        checkpoint=checkpoint,
        test_metrics=test_metrics,
    )

    print(f"Metrics CSV: {metrics_path}")
    print(f"Test metrics CSV: {test_metrics_path}")


if __name__ == "__main__":
    main()