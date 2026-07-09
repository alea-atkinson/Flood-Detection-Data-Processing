"""
Masked-Autoencoder-style self-supervised pretraining for a U-Net on UAVSAR
RGB-decomposed flood tiles.

Adapted from a repo that does patch-block MAE for gastrointestinal image
classification. Key differences from that repo:

  1. Masking operates on float32 SAR tensors (post-normalization), not
     uint8 PIL images 
  2. Patches are only eligible for masking if they are mostly *valid* SAR
     pixels (no tiles mostly outside of the flight path)
  3. Reconstruction loss is computed ONLY on masked + valid pixels (true
     MAE-style), not the whole image. The original repo computes MSE over
     every pixel, which mostly trains an identity/denoising function and
     weakens the "learn to infer from context" pressure
  4. The model is the original UNet class form the supervised learning task unmodified -- pretraining just
     instantiates it with out_channels=3 (reconstruct the 3 SAR channels)
     instead of out_channels=1 (flood logit).

USAGE
-----
    python mae_pretrain_uavsar.py \
        --train_csv /path/to/train.csv \
        --val_csv /path/to/val.csv \
        --out_dir /path/to/checkpoints \
        --patch_size 16 --mask_ratio 0.5 --epochs 200

Then, for fine-tuning on flood segmentation:

    from mae_pretrain_uavsar import load_pretrained_encoder_decoder
    seg_model = UNet(in_channels=3, out_channels=1, base_channels=32)
    load_pretrained_encoder_decoder(seg_model, "checkpoints/best_mae.pth")
"""

import argparse
import csv
import random
from pathlib import Path

import numpy as np
import rasterio
import torch
from torch import nn, optim
from torch.utils.data import DataLoader, Dataset

torch.manual_seed(98)
random.seed(98)
np.random.seed(98)


# ---------------------------------------------------------------------------
# Model -- copy of your existing architecture, unmodified. If you already
# have this in its own module, delete this class and do:
#     from your_model_file import UNet, DoubleConv
# instead.
# ---------------------------------------------------------------------------

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
    """Small, plain U-Net. Same class used for both pretraining (out_channels=3,
    reconstructing SAR bands) and segmentation fine-tuning (out_channels=1)."""

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
# Patch masking, SAR/no-data aware
# ---------------------------------------------------------------------------

def generate_patch_mask(
    valid: np.ndarray,
    patch_size: int,
    mask_ratio: float,
    min_valid_frac: float = 0.9,
) -> np.ndarray:
    """Return a boolean (H, W) array, True = "this pixel is masked out and
    should be reconstructed". Only patches that are at least `min_valid_frac`
    valid SAR pixels are eligible to be chosen for masking -- masking no-data
    regions would waste supervision on nothing.

    H and W must be divisible by patch_size. If a tile's H/W isn't divisible,
    crop/pad upstream before calling this.
    """
    h, w = valid.shape
    if h % patch_size != 0 or w % patch_size != 0:
        raise ValueError(
            f"Tile shape ({h}, {w}) is not divisible by patch_size={patch_size}. "
            "Crop or pad tiles to a multiple of patch_size."
        )

    n_rows, n_cols = h // patch_size, w // patch_size
    patch_mask = np.zeros((n_rows, n_cols), dtype=bool)

    eligible = []
    for r in range(n_rows):
        for c in range(n_cols):
            block = valid[r * patch_size:(r + 1) * patch_size, c * patch_size:(c + 1) * patch_size]
            if block.mean() >= min_valid_frac:
                eligible.append((r, c))

    num_to_mask = int(len(eligible) * mask_ratio)
    random.shuffle(eligible)
    for r, c in eligible[:num_to_mask]:
        patch_mask[r, c] = True

    # Upsample patch-level mask to full pixel resolution.
    full_mask = np.kron(patch_mask, np.ones((patch_size, patch_size), dtype=bool))
    return full_mask


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

class MAEFloodTileDataset(Dataset):
    """Reads UAVSAR tiles (reusing your exact per-tile normalization) and
    produces (masked_input, reconstruction_target, loss_mask) triples for
    MAE-style pretraining.

    CSV only needs a `uavsar_path` column -- no flood mask required, since
    this stage is fully self-supervised.
    """

    def __init__(
        self,
        csv_path: Path,
        patch_size: int = 16,
        mask_ratio: float = 0.5,
        min_valid_frac: float = 0.9,
    ) -> None:
        self.csv_path = Path(csv_path)
        self.rows = self._read_rows(self.csv_path)
        self.patch_size = patch_size
        self.mask_ratio = mask_ratio
        self.min_valid_frac = min_valid_frac

    @staticmethod
    def _read_rows(csv_path: Path) -> list[dict[str, str]]:
        with csv_path.open("r", newline="", encoding="utf-8") as handle:
            reader = csv.DictReader(handle)
            if "uavsar_path" not in (reader.fieldnames or []):
                raise ValueError(f"{csv_path} is missing required column 'uavsar_path'")
            return list(reader)

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int):
        sar_path = self.rows[index]["uavsar_path"]

        with rasterio.open(sar_path) as src:
            sar = src.read().astype(np.float32)

        if sar.shape[0] < 3:
            raise ValueError(f"Expected at least 3 SAR bands, got {sar.shape[0]} in {sar_path}")
        sar = sar[:3]

        sar_valid = ~(sar == 0).all(axis=0)
        sar[:, ~sar_valid] = np.nan
        sar = self._normalize_per_tile(sar, sar_valid)

        # Crop to a multiple of patch_size (top-left crop; swap for center-crop
        # if you'd rather not bias toward the top-left of each tile).
        h, w = sar_valid.shape
        h_crop = (h // self.patch_size) * self.patch_size
        w_crop = (w // self.patch_size) * self.patch_size
        sar = sar[:, :h_crop, :w_crop]
        sar_valid = sar_valid[:h_crop, :w_crop]

        full_mask = generate_patch_mask(
            sar_valid, self.patch_size, self.mask_ratio, self.min_valid_frac
        )

        masked_sar = sar.copy()
        masked_sar[:, full_mask] = 0.0  # matches the "invalid pixel" fill value

        # Loss is only computed where a pixel was (a) chosen for masking and
        # (b) actually valid SAR data to begin with.
        loss_mask = full_mask & sar_valid

        return (
            torch.from_numpy(masked_sar),                       # model input
            torch.from_numpy(sar),                               # reconstruction target
            torch.from_numpy(loss_mask.astype(np.float32)[None]),  # (1, H, W)
        )

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


# ---------------------------------------------------------------------------
# Masked reconstruction loss
# ---------------------------------------------------------------------------

def masked_mse_loss(
    pred: torch.Tensor, target: torch.Tensor, loss_mask: torch.Tensor, eps: float = 1e-6
) -> torch.Tensor:
    """pred, target: (B, C, H, W). loss_mask: (B, 1, H, W), 1 = include in loss."""
    diff2 = (pred - target) ** 2
    diff2 = diff2 * loss_mask  # broadcasts over channel dim
    denom = loss_mask.sum() * pred.shape[1] + eps
    return diff2.sum() / denom


# ---------------------------------------------------------------------------
# Weight transfer to the segmentation model
# ---------------------------------------------------------------------------

def load_pretrained_encoder_decoder(seg_model: nn.Module, checkpoint_path: str) -> None:
    """Loads all encoder/decoder weights from an MAE checkpoint into a
    segmentation UNet, skipping the final 1x1 conv (its out_channels differs:
    3 reconstruction channels during pretraining vs. 1 flood logit here)."""
    state = torch.load(checkpoint_path, map_location="cpu")
    if "model_state_dict" in state:
        state = state["model_state_dict"]

    filtered = {k: v for k, v in state.items() if not k.startswith("out.")}
    missing, unexpected = seg_model.load_state_dict(filtered, strict=False)
    print(f"Loaded pretrained weights. Missing keys (expected: final 'out' layer): {missing}")
    if unexpected:
        print(f"Unexpected keys ignored: {unexpected}")


# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------

def train(args: argparse.Namespace) -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    train_dataset = MAEFloodTileDataset(
        args.train_csv, patch_size=args.patch_size, mask_ratio=args.mask_ratio
    )
    val_dataset = MAEFloodTileDataset(
        args.val_csv, patch_size=args.patch_size, mask_ratio=args.mask_ratio
    )

    train_loader = DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True,
        num_workers=args.num_workers, drop_last=True,
    )
    val_loader = DataLoader(
        val_dataset, batch_size=args.batch_size, shuffle=False,
        num_workers=args.num_workers,
    )

    model = UNet(in_channels=3, out_channels=3, base_channels=args.base_channels).to(device)
    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    best_val_loss = float("inf")

    for epoch in range(args.epochs):
        model.train()
        train_loss_sum, train_batches = 0.0, 0
        for masked_sar, target_sar, loss_mask in train_loader:
            masked_sar = masked_sar.to(device)
            target_sar = target_sar.to(device)
            loss_mask = loss_mask.to(device)

            # Skip degenerate batches where nothing eligible got masked
            # (e.g. tiles that are mostly no-data).
            if loss_mask.sum() < 1:
                continue

            optimizer.zero_grad()
            pred = model(masked_sar)
            loss = masked_mse_loss(pred, target_sar, loss_mask)
            loss.backward()
            optimizer.step()

            train_loss_sum += loss.item()
            train_batches += 1

        train_loss = train_loss_sum / max(train_batches, 1)

        model.eval()
        val_loss_sum, val_batches = 0.0, 0
        with torch.no_grad():
            for masked_sar, target_sar, loss_mask in val_loader:
                masked_sar = masked_sar.to(device)
                target_sar = target_sar.to(device)
                loss_mask = loss_mask.to(device)
                if loss_mask.sum() < 1:
                    continue
                pred = model(masked_sar)
                loss = masked_mse_loss(pred, target_sar, loss_mask)
                val_loss_sum += loss.item()
                val_batches += 1

        val_loss = val_loss_sum / max(val_batches, 1)
        print(f"Epoch {epoch + 1}/{args.epochs} | train loss: {train_loss:.5f} | val loss: {val_loss:.5f}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(
                {"model_state_dict": model.state_dict(), "epoch": epoch, "val_loss": val_loss},
                out_dir / "best_mae.pth",
            )

    print(f"Done. Best val loss: {best_val_loss:.5f}. Checkpoint: {out_dir / 'best_mae.pth'}")


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="MAE-style pretraining for UAVSAR U-Net")
    parser.add_argument("--train_csv", type=Path, default="training/pretrain_csvs/train.csv")
    parser.add_argument("--val_csv", type=Path, default="training/pretrain_csvs/val.csv")
    parser.add_argument("--out_dir", type=Path, default="training/pretrain_weights")
    parser.add_argument("--patch_size", type=int, default=16)
    parser.add_argument("--mask_ratio", type=float, default=0.5)
    parser.add_argument("--base_channels", type=int, default=32)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--num_workers", type=int, default=2)
    return parser


if __name__ == "__main__":
    args = build_arg_parser().parse_args()
    train(args)