#!/usr/bin/env python3
"""Fine-tune a SAR-only U-Net for flood/change segmentation, optionally
initializing the encoder/decoder from an MAE pretraining checkpoint
(see mae_pretrain_uavsar.py).

Example (from scratch, same as before):
    python3 training/fine_tune.py \
        --train-csv {your training csv here} \
        --val-csv {your validation csv here} \
        --test-csv {your test csv here}

Example (fine-tuning from an MAE checkpoint, with a 5-epoch frozen-encoder
warm-up before unfreezing everything):
    python3 training/fine_tune.py \
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

from collections import Counter

import re

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
  --threshold F                   Probability threshold for binary prediction (default: 0.5)

Default split:
  training/fine_tune_csvs/train.csv
  training/fine_tune_csvs/val.csv
  training/fine_tune_csvs/test.csv
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
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler


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

def extract_source_fp_from_row(row: dict[str, str]) -> str:
    text = f"{row.get('uavsar_path', '')} {row.get('flood_mask_path', '')}".lower()
    match = re.search(r"(?<![a-z0-9])fp([1-7])(?![a-z0-9])", text)
    if match:
        return f"fp{match.group(1)}"
    raise ValueError(f"Could not extract source flight path from row: {row}")


def assign_flood_bin_from_fraction(frac: float) -> str:
    if frac < 0.01:
        return "0-1%"
    if frac < 0.05:
        return "1-5%"
    if frac < 0.10:
        return "5-10%"
    if frac < 0.25:
        return "10-25%"
    if frac < 0.50:
        return "25-50%"
    return ">50%"


def compute_flood_fraction_for_row(row: dict[str, str]) -> float:
    sar_path = Path(row["uavsar_path"])
    mask_path = Path(row["flood_mask_path"])

    with rasterio.open(sar_path) as src:
        sar = src.read(out_dtype="float32")[:3]

    with rasterio.open(mask_path) as src:
        mask = src.read(1, out_dtype="float32")

    valid = ~(sar == 0).all(axis=0)
    valid_pixels = int(valid.sum())

    if valid_pixels == 0:
        return 0.0

    flood_pixels = int(((mask > 0) & valid).sum())
    return flood_pixels / valid_pixels


def make_balanced_sampler(
    dataset: FloodTileDataset,
    strategy: str,
    max_weight_multiplier: float = 5.0,
) -> WeightedRandomSampler | None:
    if strategy == "standard":
        return None

    keys = []

    for row in dataset.rows:
        source_fp = extract_source_fp_from_row(row)

        if strategy == "source_fp":
            key = source_fp

        elif strategy == "flood_bin":
            flood_frac = compute_flood_fraction_for_row(row)
            key = assign_flood_bin_from_fraction(flood_frac)

        elif strategy == "source_fp_x_flood_bin":
            flood_frac = compute_flood_fraction_for_row(row)
            flood_bin = assign_flood_bin_from_fraction(flood_frac)
            key = f"{source_fp}|{flood_bin}"

        else:
            raise ValueError(f"Unknown sampling strategy: {strategy}")

        keys.append(key)

    counts = Counter(keys)

    raw_weights = np.array([1.0 / counts[key] for key in keys], dtype=np.float64)

    # Clip extreme oversampling for tiny source_fp x flood_bin groups.
    mean_weight = float(raw_weights.mean())
    max_weight = mean_weight * max_weight_multiplier
    clipped_weights = np.minimum(raw_weights, max_weight)

    print(f"Sampling strategy: {strategy}")
    print("Sampling groups:")
    for key, count in sorted(counts.items()):
        print(f"  {key}: {count}")

    return WeightedRandomSampler(
        weights=torch.as_tensor(clipped_weights, dtype=torch.double),
        num_samples=len(clipped_weights),
        replacement=True,
    )

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


def confusion_counts_from_logits(
    logits: torch.Tensor,
    targets: torch.Tensor,
    threshold: float = 0.5,
) -> dict[str, int]:
    """Return TP/FP/FN/TN counts for a batch.

    Counts are accumulated over pixels. Metrics such as Dice and IoU should
    be computed once from counts accumulated across the full split, not by
    averaging per-batch Dice/IoU values.
    """
    probs = torch.sigmoid(logits)
    preds = probs > threshold
    targets_bool = targets > 0.5

    tp = (preds & targets_bool).sum().item()
    fp = (preds & ~targets_bool).sum().item()
    fn = (~preds & targets_bool).sum().item()
    tn = (~preds & ~targets_bool).sum().item()

    return {
        "tp": int(tp),
        "fp": int(fp),
        "fn": int(fn),
        "tn": int(tn),
    }


def metrics_from_counts(
    loss: float,
    tp: int,
    fp: int,
    fn: int,
    tn: int,
) -> dict[str, float | int]:
    """Compute global segmentation metrics from accumulated pixel counts."""
    eps = 1e-7

    dice = (2.0 * tp + eps) / (2.0 * tp + fp + fn + eps)
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
    threshold: float = 0.5,
) -> dict[str, float | int]:
    is_train = optimizer is not None
    model.train(is_train)

    total_loss = 0.0
    total_batches = 0
    tp = 0
    fp = 0
    fn = 0
    tn = 0

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

        counts = confusion_counts_from_logits(
            logits.detach(),
            masks,
            threshold=threshold,
        )
        tp += counts["tp"]
        fp += counts["fp"]
        fn += counts["fn"]
        tn += counts["tn"]

        total_loss += float(loss.item())
        total_batches += 1

    if total_batches == 0:
        raise ValueError("DataLoader produced no batches.")

    avg_loss = total_loss / total_batches
    return metrics_from_counts(
        loss=avg_loss,
        tp=tp,
        fp=fp,
        fn=fn,
        tn=tn,
    )


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
            "train_accuracy",
            "train_tp",
            "train_fp",
            "train_fn",
            "train_tn",
            "val_loss",
            "val_dice",
            "val_iou",
            "val_precision",
            "val_recall",
            "val_accuracy",
            "val_tp",
            "val_fp",
            "val_fn",
            "val_tn",
            "encoder_frozen",
        ]
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_test_metrics_csv(
    test_metrics_path: Path,
    row: dict[str, float | int | str],
) -> None:
    test_metrics_path.parent.mkdir(parents=True, exist_ok=True)
    with test_metrics_path.open("w", newline="", encoding="utf-8") as handle:
        fieldnames = [
            "run_name",
            "train_csv",
            "val_csv",
            "test_csv",
            "pretrained_checkpoint",
            "best_epoch",
            "best_val_loss",
            "threshold",
            "test_loss",
            "test_dice",
            "test_iou",
            "test_precision",
            "test_recall",
            "test_accuracy",
            "test_tp",
            "test_fp",
            "test_fn",
            "test_tn",
        ]
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerow(row)


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
        "--pretrained-checkpoint", type=Path, default=None,
        help="Optional MAE checkpoint used to initialize encoder/decoder weights.",
    )
    parser.add_argument(
        "--freeze-epochs", type=int, default=0,
        help="Number of initial epochs to keep the pretrained encoder frozen before unfreezing. 0 = never frozen.",
    )
    parser.add_argument(
        "--encoder-lr-scale", type=float, default=1.0,
        help="Multiplier applied to --learning-rate for encoder/bottleneck params (once unfrozen).",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.5,
        help="Probability threshold used to convert sigmoid outputs into binary predictions.",
    )
    parser.add_argument(
    "--sampling-strategy",
    choices=["standard", "source_fp", "flood_bin", "source_fp_x_flood_bin"],
    default="standard",
    help="Training sampler strategy. Only affects the training DataLoader.",
)
    parser.add_argument(
    "--sampler-max-weight-multiplier",
    type=float,
    default=5.0,
    help="Clips sampler weights to mean_weight * this value to avoid extreme oversampling.",
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
    train_sampler = make_balanced_sampler(
        train_dataset,
        strategy=args.sampling_strategy,
        max_weight_multiplier=args.sampler_max_weight_multiplier,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=(train_sampler is None),
        sampler=train_sampler,
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

    using_pretrained = args.pretrained_checkpoint is not None

    if using_pretrained:
        load_pretrained_encoder_decoder(
            model,
            args.pretrained_checkpoint,
        )

        if args.freeze_epochs > 0:
            set_encoder_frozen(model, frozen=True)
            print(
                f"Encoder frozen for the first "
                f"{args.freeze_epochs} epoch(s)."
            )

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
    test_metrics_path = args.results_dir / f"{args.run_name}_test_metrics.csv"

    best_val_loss = float("inf")
    best_epoch = -1
    history: list[dict[str, float | int]] = []

    for epoch in range(1, args.epochs + 1):
        if (
            using_pretrained
            and args.freeze_epochs > 0
            and epoch == args.freeze_epochs + 1
        ):
            set_encoder_frozen(model, frozen=False)
            print(f"Epoch {epoch}: unfreezing encoder (lr scale={args.encoder_lr_scale}).")

        encoder_frozen = (
            using_pretrained
            and args.freeze_epochs > 0
            and epoch <= args.freeze_epochs
        )

        train_metrics = run_epoch(
            model,
            train_loader,
            loss_fn,
            device,
            optimizer,
            threshold=args.threshold,
        )
        val_metrics = run_epoch(
            model,
            val_loader,
            loss_fn,
            device,
            threshold=args.threshold,
        )

        row = {
            "epoch": epoch,
            "train_loss": train_metrics["loss"],
            "train_dice": train_metrics["dice"],
            "train_iou": train_metrics["iou"],
            "train_precision": train_metrics["precision"],
            "train_recall": train_metrics["recall"],
            "train_accuracy": train_metrics["accuracy"],
            "train_tp": train_metrics["tp"],
            "train_fp": train_metrics["fp"],
            "train_fn": train_metrics["fn"],
            "train_tn": train_metrics["tn"],
            "val_loss": val_metrics["loss"],
            "val_dice": val_metrics["dice"],
            "val_iou": val_metrics["iou"],
            "val_precision": val_metrics["precision"],
            "val_recall": val_metrics["recall"],
            "val_accuracy": val_metrics["accuracy"],
            "val_tp": val_metrics["tp"],
            "val_fp": val_metrics["fp"],
            "val_fn": val_metrics["fn"],
            "val_tn": val_metrics["tn"],
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
            best_val_loss = float(val_metrics["loss"])
            best_epoch = epoch
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
    best_epoch = int(checkpoint.get("epoch", best_epoch))
    best_val_loss = float(checkpoint.get("best_val_loss", best_val_loss))

    test_metrics = run_epoch(
        model,
        test_loader,
        loss_fn,
        device,
        threshold=args.threshold,
    )

    test_row = {
        "run_name": args.run_name,
        "train_csv": args.train_csv.as_posix(),
        "val_csv": args.val_csv.as_posix(),
        "test_csv": args.test_csv.as_posix(),
        "pretrained_checkpoint": (
            args.pretrained_checkpoint.as_posix()
            if args.pretrained_checkpoint is not None
            else ""
        ),
        "best_epoch": best_epoch,
        "best_val_loss": best_val_loss,
        "threshold": args.threshold,
        "test_loss": test_metrics["loss"],
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
    write_test_metrics_csv(test_metrics_path, test_row)

    print(
        "Best checkpoint test metrics: "
        f"loss={test_metrics['loss']:.4f} "
        f"dice={test_metrics['dice']:.4f} "
        f"iou={test_metrics['iou']:.4f} "
        f"precision={test_metrics['precision']:.4f} "
        f"recall={test_metrics['recall']:.4f} "
        f"tp={test_metrics['tp']} "
        f"fp={test_metrics['fp']} "
        f"fn={test_metrics['fn']} "
        f"tn={test_metrics['tn']}"
    )
    print(f"Training metrics CSV: {metrics_path}")
    print(f"Test metrics CSV: {test_metrics_path}")


if __name__ == "__main__":
    main()