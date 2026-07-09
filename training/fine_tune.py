#!/usr/bin/env python3
"""Fine-tune a SAR-only U-Net for flood/change segmentation, optionally
initializing the encoder/decoder from an MAE pretraining checkpoint
(see mae_pretrain_uavsar.py).

Example (from scratch, same as before):
    python3 train_unet_finetune.py \
        --train-csv {your training csv here} \
        --val-csv {your validation csv here} \
        --test-csv {your test csv here}

Example (fine-tuning from an MAE checkpoint, with a 5-epoch frozen-encoder
warm-up before unfreezing everything):
    python3 train_unet_finetune.py \
        --train-csv {your training csv here} \
        --val-csv {your validation csv here} \
        --test-csv {your test csv here} \
        --pretrained-checkpoint checkpoints/best_mae.pth \
        --freeze-epochs 5 \
        --encoder-lr-scale 0.3
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
        """usage: train_unet_finetune.py [options]

Fine-tune a SAR-only binary U-Net, optionally from an MAE checkpoint.

options:
  --data-root PATH             Root folder for tile paths (default: 2025_Tile_Data)
  --train-csv PATH             Training split CSV
  --val-csv PATH               Validation split CSV
  --test-csv PATH              Test split CSV
  --epochs N                   Number of epochs (default: 20)
  --batch-size N                Batch size (default: 16)
  --learning-rate LR           Adam learning rate for decoder/new layers (default: 9.33e-05)
  --weight-decay WD            AdamW weight decay (default: 6.09e-06)
  --num-workers N               DataLoader workers (default: 2)
  --base-channels N             U-Net width (default: 32)
  --seed N                      Random seed (default: 42)
  --models-dir PATH             Checkpoint output folder (default: models)
  --results-dir PATH            Metrics CSV output folder (default: results)
  --run-name NAME                Prefix for output files
  --device DEVICE                cuda or cpu
  --pretrained-checkpoint PATH  MAE checkpoint to initialize encoder/decoder from (optional)
  --freeze-epochs N              Epochs to keep the pretrained encoder frozen before
                                  unfreezing (default: 0, meaning no freezing)
  --encoder-lr-scale F            Multiplier applied to --learning-rate for encoder
                                  params once unfrozen (default: 1.0)

Default split:
  strict_no_overlap/heldout_fp1_train.csv
  strict_no_overlap/heldout_fp1_validation.csv
  strict_no_overlap/heldout_fp1_test.csv
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

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        row = self.rows[index]
        sar_path = row["uavsar_path"]
        mask_path = row["flood_mask_path"]

        with rasterio.open(sar_path) as src:
            sar = src.read(out_dtype="float32")
        if sar.shape[0] < 3:
            raise ValueError(f"Expected at least 3 SAR bands, got {sar.shape[0]} in {sar_path}")
        sar = sar[:3]

        sar_valid = ~(sar == 0).all(axis=0)
        sar[:, ~sar_valid] = np.nan
        sar = self._normalize_per_tile(sar, sar_valid)

        with rasterio.open(mask_path) as src:
            mask = src.read(1, out_dtype="float32")

        mask = (mask > 0).astype(np.float32)[None, :, :]

        return torch.from_numpy(sar), torch.from_numpy(mask)

    @staticmethod
    def _normalize_per_tile(
        sar: np.ndarray,
        valid: np.ndarray,
    ) -> np.ndarray:
 
        sar = sar.copy()
 
        for c in range(sar.shape[0]):
 
            band = sar[c]
 
            values = band[valid]
 
            if values.size == 0:
                band[:] = 0.0
                sar[c] = band
                continue
 
            low, high = np.percentile(values, [1.0, 99.0])
 
            values = np.clip(values, low, high)
 
            mean = values.mean()
            std = values.std()
 
            if std < 1e-6:
                band[:] = 0.0
            else:
                band[valid] = (values - mean) / std
 
            # Keep invalid pixels at zero
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
    """Small, plain U-Net for binary segmentation."""

    def __init__(self, in_channels: int = 3, out_channels: int = 1, base_channels: int = 32) -> None:
        super().__init__()
        self.enc1 = DoubleConv(in_channels, base_channels)
        self.enc2 = DoubleConv(base_channels, base_channels * 2)
        self.enc3 = DoubleConv(base_channels * 2, base_channels * 4)
        self.enc4 = DoubleConv(base_channels * 4, base_channels * 8)

        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.bottleneck = DoubleConv(base_channels * 8, base_channels * 16)

        self.up4 = nn.ConvTranspose2d(base_channels * 16, base_channels * 8, kernel_size=2, stride=2)
        self.dec4 = DoubleConv(base_channels * 16, base_channels * 8)
        self.up3 = nn.ConvTranspose2d(base_channels * 8, base_channels * 4, kernel_size=2, stride=2)
        self.dec3 = DoubleConv(base_channels * 8, base_channels * 4)
        self.up2 = nn.ConvTranspose2d(base_channels * 4, base_channels * 2, kernel_size=2, stride=2)
        self.dec2 = DoubleConv(base_channels * 4, base_channels * 2)
        self.up1 = nn.ConvTranspose2d(base_channels * 2, base_channels, kernel_size=2, stride=2)
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


# ---------------------------------------------------------------------------
# Pretrained weight loading + encoder freeze/unfreeze support
# ---------------------------------------------------------------------------

ENCODER_PREFIXES = ("enc1", "enc2", "enc3", "enc4", "bottleneck")


def is_encoder_param(name: str) -> bool:
    return name.startswith(ENCODER_PREFIXES)


def load_pretrained_encoder_decoder(model: nn.Module, checkpoint_path: Path) -> None:
    """Loads encoder/decoder weights from an MAE checkpoint, skipping the
    final 1x1 'out' conv (its channel count differs: 3 reconstruction
    channels during pretraining vs. 1 flood logit here)."""
    state = torch.load(checkpoint_path, map_location="cpu")
    if "model_state_dict" in state:
        state = state["model_state_dict"]

    filtered = {k: v for k, v in state.items() if not k.startswith("out.")}
    missing, unexpected = model.load_state_dict(filtered, strict=False)
    print(f"Loaded pretrained checkpoint: {checkpoint_path}")
    print(f"  Missing keys (expected: only the final 'out' layer): {missing}")
    if unexpected:
        print(f"  Unexpected keys ignored: {unexpected}")


def set_encoder_frozen(model: nn.Module, frozen: bool) -> None:
    for name, param in model.named_parameters():
        if is_encoder_param(name):
            param.requires_grad = not frozen


def dice_iou_from_logits(logits: torch.Tensor, targets: torch.Tensor, threshold: float = 0.5) -> tuple[float, float]:
    probs = torch.sigmoid(logits)
    preds = probs > threshold
    targets_bool = targets > 0.5

    intersection = (preds & targets_bool).sum().float()
    pred_sum = preds.sum().float()
    target_sum = targets_bool.sum().float()
    union = (preds | targets_bool).sum().float()

    eps = torch.tensor(1e-7, device=logits.device)
    dice = (2.0 * intersection + eps) / (pred_sum + target_sum + eps)
    iou = (intersection + eps) / (union + eps)
    return float(dice.item()), float(iou.item())


#CUSTOM LOSS FUNCTIONS 

class DiceLoss(nn.Module):
    def __init__(self, smooth=1.0):
        super().__init__()
        self.smooth = smooth

    def forward(self, logits, targets):

        probs = torch.sigmoid(logits)

        probs = probs.view(-1)
        targets = targets.view(-1)

        intersection = (probs * targets).sum()

        dice = (
            2.0 * intersection + self.smooth
        ) / (
            probs.sum() + targets.sum() + self.smooth
        )

        return 1 - dice


class BCEDiceLoss(nn.Module):
    def __init__(self):
        super().__init__()

        self.bce = nn.BCEWithLogitsLoss()
        self.dice = DiceLoss()

    def forward(self, logits, targets):

        bce_loss = self.bce(logits, targets)
        dice_loss = self.dice(logits, targets)

        return bce_loss + dice_loss
    
#handles class imbalance by downweighting easy negatives and focusing more on hard examples

class FocalLoss(nn.Module):
    def __init__(self, alpha=0.25, gamma=2.0):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, logits, targets):

        bce = F.binary_cross_entropy_with_logits(
            logits,
            targets,
            reduction="none"
        )

        probs = torch.sigmoid(logits)

        pt = torch.where(
            targets == 1,
            probs,
            1 - probs
        )

        focal_weight = self.alpha * (1 - pt) ** self.gamma

        loss = focal_weight * bce

        return loss.mean()
    
#focal and dice combined to handle class imbalance and optimize for segmentation metrics

class FocalDiceLoss(nn.Module):
    def __init__(
        self,
        alpha=0.25,
        gamma=2.0,
        dice_weight=1.0,
        focal_weight=1.0,
    ):
        super().__init__()

        self.focal = FocalLoss(alpha, gamma)
        self.dice = DiceLoss()

        self.dice_weight = dice_weight
        self.focal_weight = focal_weight

    def forward(self, logits, targets):

        focal_loss = self.focal(logits, targets)

        dice_loss = self.dice(logits, targets)

        return (
            self.focal_weight * focal_loss
            + self.dice_weight * dice_loss
        )


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
    total_dice = 0.0
    total_iou = 0.0
    total_batches = 0

    for images, masks in loader:
        images = images.to(device, non_blocking=True)
        masks = masks.to(device, non_blocking=True)

        with torch.set_grad_enabled(is_train):
            logits = model(images)
            loss = loss_fn(logits, masks)
            if is_train:
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step()

        dice, iou = dice_iou_from_logits(logits.detach(), masks)
        total_loss += float(loss.item())
        total_dice += dice
        total_iou += iou
        total_batches += 1

    if total_batches == 0:
        raise ValueError("DataLoader produced no batches.")

    return {
        "loss": total_loss / total_batches,
        "dice": total_dice / total_batches,
        "iou": total_iou / total_batches,
    }


def write_metrics_csv(metrics_path: Path, rows: list[dict[str, float | int]]) -> None:
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    with metrics_path.open("w", newline="", encoding="utf-8") as handle:
        fieldnames = ["epoch", "train_loss", "train_dice", "train_iou", "val_loss", "val_dice", "val_iou", "encoder_frozen"]
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Fine-tune a SAR-only binary U-Net, optionally from an MAE checkpoint.")
    parser.add_argument("--data-root", type=Path, default=Path("2025_Tile_Data"))
    parser.add_argument("--train-csv", type=Path, default="training/fine_tune_csvs/train.csv")
    parser.add_argument("--val-csv", type=Path, default="training/fine_tune_csvs/val.csv")
    parser.add_argument("--test-csv", type=Path, default="training/fine_tune_csvs/test.csv")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--learning-rate", type=float, default= 9.327106954111342e-05)
    parser.add_argument("--weight-decay", type=float, default=  6.088353841746043e-06)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--base-channels", type=int, default=32)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--models-dir", type=Path, default=Path("models"))
    parser.add_argument("--results-dir", type=Path, default=Path("results"))
    parser.add_argument("--run-name", default="unet_baseline_strict_tuned_fp1")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument(
        "--pretrained-checkpoint", type=Path, default="training/pretrain_weights/best_mae.pth",
        help="MAE checkpoint (from mae_pretrain_uavsar.py) to initialize encoder/decoder from.",
    )
    parser.add_argument(
        "--freeze-epochs", type=int, default=5,
        help="Number of initial epochs to keep the pretrained encoder frozen before unfreezing. 0 = never frozen.",
    )
    parser.add_argument(
        "--encoder-lr-scale", type=float, default=0.01,
        help="Multiplier applied to --learning-rate for encoder/bottleneck params (once unfrozen).",
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
    print(f"Dataset sizes: train={len(train_dataset)}, val={len(val_dataset)}, test={len(test_dataset)}")

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

    model = UNet(in_channels=3, out_channels=1, base_channels=args.base_channels).to(device)

    if args.pretrained_checkpoint is not None:
        load_pretrained_encoder_decoder(model, args.pretrained_checkpoint)
        if args.freeze_epochs > 0:
            set_encoder_frozen(model, frozen=True)
            print(f"Encoder frozen for the first {args.freeze_epochs} epoch(s).")

    loss_fn = FocalDiceLoss()

    # Separate param groups so the pretrained encoder can use a different
    # (typically lower) LR than the randomly-initialized decoder/out layer
    # once it's unfrozen. With --freeze-epochs 0 and --encoder-lr-scale 1.0
    # this behaves identically to a single-LR optimizer.
    encoder_params = [p for n, p in model.named_parameters() if is_encoder_param(n)]
    decoder_params = [p for n, p in model.named_parameters() if not is_encoder_param(n)]
    optimizer = torch.optim.AdamW(
        [
            {"params": encoder_params, "lr": args.learning_rate * args.encoder_lr_scale},
            {"params": decoder_params, "lr": args.learning_rate},
        ],
        weight_decay=args.weight_decay,
    )

    args.models_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = args.models_dir / f"{args.run_name}_best.pt"
    metrics_path = args.results_dir / f"{args.run_name}_metrics.csv"

    best_val_loss = float("inf")
    history: list[dict[str, float | int]] = []

    for epoch in range(1, args.epochs + 1):
        if args.freeze_epochs > 0 and epoch == args.freeze_epochs + 1:
            set_encoder_frozen(model, frozen=False)
            print(f"Epoch {epoch}: unfreezing encoder (lr scale={args.encoder_lr_scale}).")

        encoder_frozen = args.freeze_epochs > 0 and epoch <= args.freeze_epochs

        train_metrics = run_epoch(model, train_loader, loss_fn, device, optimizer)
        val_metrics = run_epoch(model, val_loader, loss_fn, device)

        row = {
            "epoch": epoch,
            "train_loss": train_metrics["loss"],
            "train_dice": train_metrics["dice"],
            "train_iou": train_metrics["iou"],
            "val_loss": val_metrics["loss"],
            "val_dice": val_metrics["dice"],
            "val_iou": val_metrics["iou"],
            "encoder_frozen": int(encoder_frozen),
        }
        history.append(row)
        write_metrics_csv(metrics_path, history)

        print(
            f"Epoch {epoch:03d}/{args.epochs:03d} "
            f"train_loss={train_metrics['loss']:.4f} "
            f"val_loss={val_metrics['loss']:.4f} "
            f"val_dice={val_metrics['dice']:.4f} "
            f"val_iou={val_metrics['iou']:.4f} "
            f"{'[encoder frozen]' if encoder_frozen else ''}"
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
    weights_only=False
    )
    model.load_state_dict(checkpoint["model_state_dict"])
    test_metrics = run_epoch(model, test_loader, loss_fn, device)
    print(
        "Best checkpoint test metrics: "
        f"loss={test_metrics['loss']:.4f} "
        f"dice={test_metrics['dice']:.4f} "
        f"iou={test_metrics['iou']:.4f}"
    )
    print(f"Metrics CSV: {metrics_path}")


if __name__ == "__main__":
    main()