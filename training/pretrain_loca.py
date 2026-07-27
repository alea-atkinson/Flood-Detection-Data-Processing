"""
Masked-Autoencoder-style self-supervised pretraining for a U-Net on UAVSAR
RGB-decomposed flood tiles.

Adapted from a repo that does patch-block MAE for gastrointestinal image
classification. Key differences from that repo:

  1. Masking operates on float32 SAR tensors (post-normalization), not
     uint8 PIL images 
  2. Patches are only eligible for masking if they are mostly *valid* SAR
     pixels (no tiles mostly outside of the flight path)
  3. Reconstruction loss is computed ONLY on masked + valid pixels 
     , not the whole image. The original repo computes MSE over
     every pixel, which mostly trains an identity/denoising function and
     weakens the "learn to infer from context" pressure. Also uses MAE for application to SAR instead of MSE
  4. The model is the original UNet class form the supervised learning task unmodified -- pretraining just
     instantiates it with out_channels=3 (reconstruct the 3 SAR channels)
     instead of out_channels=1 (flood logit).
  5. LOCA (location-aware position prediction) is optionally trained as a
     SECOND, auxiliary pretext task alongside MAE reconstruction. This is a
     CNN-adapted version of LOCA: since the U-Net has no patch tokens to
     shuffle the way a ViT does, the pretext task is implemented as a
     jigsaw-style block shuffle -- the clean (unmasked) tile is chopped
     into blocks, the blocks are randomly permuted, and a small auxiliary
     head attached to the shared encoder's bottleneck features has to
     predict each block's original grid position. This gives the encoder
     explicit localization/spatial-reasoning pressure that pure pixel
     reconstruction doesn't provide, without requiring a second dataset
     or a ViT rewrite. See the module-level LOCA section below.

USAGE
-----
    python mae_pretrain_uavsar.py \
        --train_csv /path/to/train.csv \
        --val_csv /path/to/val.csv \
        --out_dir /path/to/checkpoints \
        --patch_size 16 --mask_ratio 0.5 --epochs 200 \
        --use_loca --loca_block_size 32 --lambda_pos 0.3

Then, for fine-tuning on flood segmentation:

    from mae_pretrain_uavsar import load_pretrained_encoder_decoder
    seg_model = UNet(in_channels=3, out_channels=1, base_channels=32)
    load_pretrained_encoder_decoder(seg_model, "checkpoints/best_mae.pth")

(The position head, if LOCA was used, is saved alongside the reconstruction
weights in the checkpoint but is never loaded into the segmentation model --
`load_pretrained_encoder_decoder` only pulls encoder/decoder weights, and
`position_head.*` keys are reported as "unexpected keys ignored", same as
the final `out.*` layer.)
"""

import argparse
import csv
import random
from pathlib import Path

import numpy as np
import rasterio
import torch
import torch.nn.functional as F
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

    def forward_bottleneck(self, x: torch.Tensor) -> torch.Tensor:
        """Runs only the encoder + bottleneck (no decoder, no skip
        concatenation). Used by the LOCA position-prediction auxiliary
        task below, which only needs a spatially-pooled feature map to
        attach a lightweight classification head to -- it doesn't need a
        full pixel-wise reconstruction the way the MAE branch does.
        Shares all encoder weights with `forward`, so gradients from the
        position-prediction loss flow into the same encoder that MAE
        reconstruction trains."""
        enc1 = self.enc1(x)
        enc2 = self.enc2(self.pool(enc1))
        enc3 = self.enc3(self.pool(enc2))
        enc4 = self.enc4(self.pool(enc3))
        return self.bottleneck(self.pool(enc4))


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
# LOCA: location-aware position prediction (auxiliary pretext task)
# ---------------------------------------------------------------------------
#
# Full LOCA (as used with ViTs) samples a query/reference crop pair and has
# query patch tokens predict their position in the reference view. There's
# no token sequence here -- this is a CNN -- so this is a jigsaw-style
# adaptation: chop the clean tile into `loca_block_size`-sized blocks,
# randomly permute a subset of them (only "eligible", mostly-valid blocks --
# same philosophy as `generate_patch_mask` above, to avoid wasting the task
# on no-data regions), run the shuffled tile through the shared encoder, and
# have a small head predict each block's ORIGINAL grid position from the
# bottleneck features at that spot. This gives the encoder an explicit
# incentive to reason about spatial layout, which plain pixel reconstruction
# doesn't directly train for.
#
# Runs as a second forward pass through `model.forward_bottleneck` (weights
# shared with the MAE branch), so it adds compute but no new parameters
# beyond the small `PositionHead`.

def compute_block_validity(
    valid_mask: torch.Tensor, block_size: int, min_valid_frac: float
) -> torch.Tensor:
    """
    valid_mask: (B, H, W) bool -- True = valid SAR pixel.
    Returns: (B, n_rows, n_cols) bool -- True = block is eligible for
             LOCA shuffling (i.e. mostly valid, not mostly no-data).
    """
    b, h, w = valid_mask.shape
    n_rows, n_cols = h // block_size, w // block_size
    valid_f = valid_mask.float().view(b, n_rows, block_size, w)
    valid_f = valid_f.view(b, n_rows, block_size, n_cols, block_size)
    frac = valid_f.mean(dim=(2, 4))  # (B, n_rows, n_cols)
    return frac >= min_valid_frac


def shuffle_patches(
    x: torch.Tensor, block_size: int, eligible: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Randomly permute block_size x block_size blocks of `x`, restricted to
    the blocks flagged `eligible` (ineligible blocks -- e.g. mostly no-data
    -- are left in place and excluded from the loss).

    Args:
        x: (B, C, H, W), H and W divisible by block_size.
        block_size: side length of each shuffled block.
        eligible: (B, n_rows, n_cols) bool, which blocks may participate.

    Returns:
        shuffled: (B, C, H, W) -- x with eligible blocks permuted.
        position_labels: (B, n_rows, n_cols) long -- for each grid slot,
            the ORIGINAL flat index of the block content now sitting there.
            This is the classification target for PositionHead.
        loss_mask: (B, n_rows, n_cols) bool -- True where the block was
            actually shuffled (i.e. where the label is meaningful).
    """
    b, c, h, w = x.shape
    if h % block_size != 0 or w % block_size != 0:
        raise ValueError(
            f"Tile shape ({h}, {w}) is not divisible by loca_block_size={block_size}."
        )
    n_rows, n_cols = h // block_size, w // block_size
    n_blocks = n_rows * n_cols

    blocks = (
        x.view(b, c, n_rows, block_size, n_cols, block_size)
         .permute(0, 2, 4, 1, 3, 5)
         .reshape(b, n_blocks, c, block_size, block_size)
    )
    eligible_flat = eligible.reshape(b, n_blocks)

    shuffled_blocks = blocks.clone()
    position_labels = torch.arange(n_blocks, device=x.device).unsqueeze(0).repeat(b, 1).clone()
    loss_mask = torch.zeros(b, n_blocks, dtype=torch.bool, device=x.device)

    for i in range(b):
        idx = eligible_flat[i].nonzero(as_tuple=True)[0]
        if idx.numel() < 2:
            continue  # nothing meaningful to shuffle for this tile
        perm = idx[torch.randperm(idx.numel(), device=x.device)]
        shuffled_blocks[i, idx] = blocks[i, perm]
        position_labels[i, idx] = perm
        loss_mask[i, idx] = True

    shuffled = (
        shuffled_blocks.view(b, n_rows, n_cols, c, block_size, block_size)
        .permute(0, 3, 1, 4, 2, 5)
        .reshape(b, c, h, w)
    )
    position_labels = position_labels.view(b, n_rows, n_cols)
    loss_mask = loss_mask.view(b, n_rows, n_cols)
    return shuffled, position_labels, loss_mask


class PositionHead(nn.Module):
    """Predicts, for each grid slot in a LOCA-shuffled tile, the original
    flat index of the block now sitting there. Operates on the U-Net
    bottleneck features (shared with MAE reconstruction) via an
    AdaptiveAvgPool2d down to the block grid resolution, so it works
    regardless of the exact ratio between loca_block_size and the encoder's
    16x bottleneck downsampling."""

    def __init__(self, in_channels: int, n_rows: int, n_cols: int) -> None:
        super().__init__()
        n_blocks = n_rows * n_cols
        self.pool = nn.AdaptiveAvgPool2d((n_rows, n_cols))
        self.mlp = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels, n_blocks, kernel_size=1),
        )

    def forward(self, bottleneck_feats: torch.Tensor) -> torch.Tensor:
        feats = self.pool(bottleneck_feats)
        return self.mlp(feats)  # (B, n_blocks, n_rows, n_cols)


def position_loss_fn(
    logits: torch.Tensor, labels: torch.Tensor, loss_mask: torch.Tensor
) -> torch.Tensor:
    """
    logits: (B, n_blocks, n_rows, n_cols)
    labels: (B, n_rows, n_cols) long
    loss_mask: (B, n_rows, n_cols) bool -- which grid slots to score.
    """
    b, n_blocks, n_rows, n_cols = logits.shape
    logits_flat = logits.permute(0, 2, 3, 1).reshape(-1, n_blocks)
    labels_flat = labels.reshape(-1)
    mask_flat = loss_mask.reshape(-1)

    if mask_flat.sum() < 1:
        return torch.zeros((), device=logits.device, requires_grad=True)

    return F.cross_entropy(logits_flat[mask_flat], labels_flat[mask_flat])


def compare_encoder_grad_norms(
    model: nn.Module, recon_loss: torch.Tensor, pos_loss: torch.Tensor, lambda_pos: float
) -> tuple[float, float]:
    """Diagnostic only -- answers "is lambda_pos * pos_loss actually moving
    the shared encoder, or is it negligible next to recon_loss?" Loss
    *magnitude* comparisons can be misleading (same scale, very different
    effect on weights), so this measures gradient *norm* on the encoder/
    bottleneck parameters that both losses touch.

    Runs two extra backward passes, so it's too expensive to call every
    step -- call it every N batches (see --diagnose_every) or just a
    handful of times total. Leaves model.grad in the state of the LAST
    backward call here (pos_loss's), so callers must re-zero grad and do
    their real (combined) backward + optimizer.step() afterward -- this
    function does not consume the graph for retain_graph=False users
    since it's called with retain_graph=True on both.
    """
    encoder_params = [
        p for n, p in model.named_parameters()
        if n.startswith(("enc1", "enc2", "enc3", "enc4", "bottleneck")) and p.requires_grad
    ]

    model.zero_grad()
    recon_loss.backward(retain_graph=True)
    recon_grad_norm = sum(
        p.grad.detach().norm().item() ** 2 for p in encoder_params if p.grad is not None
    ) ** 0.5

    model.zero_grad()
    (lambda_pos * pos_loss).backward(retain_graph=True)
    pos_grad_norm = sum(
        p.grad.detach().norm().item() ** 2 for p in encoder_params if p.grad is not None
    ) ** 0.5

    model.zero_grad()
    return recon_grad_norm, pos_grad_norm


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
# loss functions
# ---------------------------------------------------------------------------

import torch


def masked_mse_loss(
    pred: torch.Tensor,
    target: torch.Tensor,
    loss_mask: torch.Tensor,
    eps: float = 1e-6,
) -> torch.Tensor:
    """
    Compute masked Mean Squared Error (MSE) loss.

    Args:
        pred: Predicted tensor of shape (B, C, H, W).
        target: Ground truth tensor of shape (B, C, H, W).
        loss_mask: Binary mask of shape (B, 1, H, W),
                   where 1 indicates pixels to include in the loss.
        eps: Small constant to avoid division by zero.

    Returns:
        Scalar masked MSE loss.
    """
    # Squared error
    diff2 = (pred - target) ** 2

    # Apply mask (broadcasts over channel dimension)
    diff2 = diff2 * loss_mask

    # Normalize by the number of valid pixels × channels
    denom = loss_mask.sum() * pred.shape[1] + eps

    return diff2.sum() / denom

def masked_mae_loss(
    pred: torch.Tensor, target: torch.Tensor, loss_mask: torch.Tensor, eps: float = 1e-6
) -> torch.Tensor:
    """
    pred, target: (B, C, H, W)
    loss_mask: (B, 1, H, W), 1 = include in loss.
    """
    diff = torch.abs(pred - target)

    # Apply mask (broadcasts over channel dimension)
    diff = diff * loss_mask

    # Normalize by number of valid pixels and channels
    denom = loss_mask.sum() * pred.shape[1] + eps

    return diff.sum() / denom

from pytorch_msssim import ssim

def masked_mae_ssim_loss(
    pred,
    target,
    loss_mask,
    alpha=0.8,
):
    # MAE only on masked pixels
    mae = masked_mae_loss(pred, target, loss_mask)

    # SSIM over the reconstructed image
    ssim_loss = 1.0 - ssim(
        pred,
        target,
        data_range=1.0,
        size_average=True,
    )

    return alpha * mae + (1 - alpha) * ssim_loss


# ---------------------------------------------------------------------------
# Weight transfer to the segmentation model
# ---------------------------------------------------------------------------

def load_pretrained_encoder_decoder(seg_model: nn.Module, checkpoint_path: str) -> None:
    """Loads all encoder/decoder weights from an MAE checkpoint into a
    segmentation UNet, skipping the final 1x1 conv (its out_channels differs:
    3 reconstruction channels during pretraining vs. 1 flood logit here).
    If the checkpoint was trained with LOCA, its `position_head.*` weights
    are also present but get silently reported as unexpected/ignored here,
    same as the `out.*` layer -- the position head has no use downstream."""
    state = torch.load(checkpoint_path, map_location="cpu")
    if "model_state_dict" in state:
        state = state["model_state_dict"]

    filtered = {k: v for k, v in state.items() if not k.startswith("out.")}
    missing, unexpected = seg_model.load_state_dict(filtered, strict=False)
    print(f"Loaded pretrained weights. Missing keys (expected: final 'out' layer): {missing}")
    if unexpected:
        print(f"Unexpected keys ignored: {unexpected}")

import matplotlib.pyplot as plt
from pathlib import Path

def display_rgb(img):
    img = img.copy()
    img -= img.min()
    img /= img.max() + 1e-8

    return img

def visualize_reconstruction(
    masked_input,
    target,
    prediction,
    loss_mask,
    save_path=None,
):
    """
    masked_input : (3,H,W)
    target       : (3,H,W)
    prediction   : (3,H,W)
    loss_mask    : (1,H,W)
    """

    masked_input = masked_input.cpu().numpy()
    target = target.cpu().numpy()
    prediction = prediction.detach().cpu().numpy()
    loss_mask = loss_mask.cpu().numpy()[0]

    # Use first three channels as RGB
    inp = np.moveaxis(masked_input, 0, -1)
    tgt = np.moveaxis(target, 0, -1)
    pred = np.moveaxis(prediction, 0, -1)

    error = np.mean(np.abs(pred - tgt), axis=-1)

    fig, ax = plt.subplots(1, 5, figsize=(22, 5))


    ax[0].imshow(display_rgb(tgt))
    ax[0].set_title("Original")

    ax[1].imshow(display_rgb(inp))
    ax[1].set_title("Masked Input")

    ax[2].imshow(display_rgb(pred))
    ax[2].set_title("Reconstruction")

    ax[3].imshow(error, cmap="gray")
    ax[3].set_title("Absolute Error")

    ax[4].imshow(loss_mask, cmap="gray")
    ax[4].set_title("Masked Pixels")

    for a in ax:
        a.axis("off")

    plt.tight_layout()

    if save_path is not None:
        plt.savefig(save_path, dpi=200)

    plt.close()


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

    # --- LOCA setup -----------------------------------------------------
    # Position head is sized off the first training tile's shape, so it
    # assumes all tiles share the same H/W (true of this pipeline's fixed
    # tile grid). It's created eagerly, before the optimizer, so its
    # parameters get included in the same optimizer as the rest of the model.
    position_head = None
    if args.use_loca:
        _, sample_target, _ = train_dataset[0]
        _, tile_h, tile_w = sample_target.shape
        if tile_h % args.loca_block_size != 0 or tile_w % args.loca_block_size != 0:
            raise ValueError(
                f"Tile shape ({tile_h}, {tile_w}) is not divisible by "
                f"--loca_block_size={args.loca_block_size}. Pick a block size "
                "that evenly divides your tile dimensions."
            )
        n_rows, n_cols = tile_h // args.loca_block_size, tile_w // args.loca_block_size
        position_head = PositionHead(
            in_channels=args.base_channels * 16, n_rows=n_rows, n_cols=n_cols
        ).to(device)
        print(
            f"LOCA enabled: block_size={args.loca_block_size}, "
            f"grid={n_rows}x{n_cols} ({n_rows * n_cols} blocks), "
            f"lambda_pos={args.lambda_pos} (warmup over {args.lambda_pos_warmup_epochs} epochs)"
        )

    params = list(model.parameters())
    if position_head is not None:
        params += list(position_head.parameters())
    optimizer = optim.Adam(params, lr=args.lr)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    best_val_loss = float("inf")

    for epoch in range(args.epochs):
        model.train()
        if position_head is not None:
            position_head.train()

        current_lambda_pos = args.lambda_pos * min(
            1.0, (epoch + 1) / max(args.lambda_pos_warmup_epochs, 1)
        )

        train_recon_sum, train_pos_sum, train_batches = 0.0, 0.0, 0
        for batch_idx, (masked_sar, target_sar, loss_mask) in enumerate(train_loader):
            masked_sar = masked_sar.to(device)
            target_sar = target_sar.to(device)
            loss_mask = loss_mask.to(device)

            # Skip degenerate batches where nothing eligible got masked
            # (e.g. tiles that are mostly no-data).
            if loss_mask.sum() < 1:
                continue

            optimizer.zero_grad()
            pred = model(masked_sar)
            if(args.loss_fn == "MSE"):
                recon_loss = masked_mse_loss(pred, target_sar, loss_mask)
            elif(args.loss_fn == "MAE"):
                recon_loss = masked_mae_loss(pred, target_sar, loss_mask)
            elif(args.loss_fn == "ssim"):
                recon_loss = masked_mae_ssim_loss(pred, target_sar, loss_mask)

            pos_loss = torch.zeros((), device=device)
            if position_head is not None:
                # Approximate per-pixel validity from the clean target: invalid
                # pixels are explicitly zeroed across all channels by
                # _normalize_per_tile, so "not all-zero across channels" is a
                # reasonable stand-in without threading sar_valid through the
                # Dataset return signature.
                approx_valid = ~(target_sar == 0).all(dim=1)  # (B, H, W)
                eligible = compute_block_validity(
                    approx_valid, args.loca_block_size, args.loca_min_valid_frac
                )
                shuffled, position_labels, block_loss_mask = shuffle_patches(
                    target_sar, args.loca_block_size, eligible
                )
                bottleneck_feats = model.forward_bottleneck(shuffled)
                pos_logits = position_head(bottleneck_feats)
                pos_loss = position_loss_fn(pos_logits, position_labels, block_loss_mask)

            if (
                position_head is not None
                and args.diagnose_grad_norms
                and batch_idx % args.diagnose_every == 0
            ):
                recon_gn, pos_gn = compare_encoder_grad_norms(
                    model, recon_loss, pos_loss, current_lambda_pos
                )
                ratio = pos_gn / (recon_gn + 1e-8)
                print(
                    f"  [diag] epoch {epoch + 1} batch {batch_idx}: "
                    f"encoder grad norm recon={recon_gn:.5f} "
                    f"pos(weighted)={pos_gn:.5f} ratio={ratio:.4f}"
                )
                # compare_encoder_grad_norms leaves model grads zeroed, so the
                # real backward below still starts from a clean slate.

            loss = recon_loss + current_lambda_pos * pos_loss
            loss.backward()
            optimizer.step()

            train_recon_sum += recon_loss.item()
            train_pos_sum += pos_loss.item()
            train_batches += 1

        train_recon_loss = train_recon_sum / max(train_batches, 1)
        train_pos_loss = train_pos_sum / max(train_batches, 1)

        model.eval()
        if position_head is not None:
            position_head.eval()

        val_loss_sum, val_batches = 0.0, 0
        with torch.no_grad():
            for masked_sar, target_sar, loss_mask in val_loader:
                masked_sar = masked_sar.to(device)
                target_sar = target_sar.to(device)
                loss_mask = loss_mask.to(device)
                if loss_mask.sum() < 1:
                    continue
                pred = model(masked_sar)
                loss = masked_mae_loss(pred, target_sar, loss_mask)
                val_loss_sum += loss.item()
                val_batches += 1

        val_loss = val_loss_sum / max(val_batches, 1)
        if position_head is not None:
            print(
                f"Epoch {epoch + 1}/{args.epochs} | "
                f"train recon: {train_recon_loss:.5f} | train pos: {train_pos_loss:.5f} "
                f"(lambda={current_lambda_pos:.3f}) | val recon: {val_loss:.5f}"
            )
        else:
            print(f"Epoch {epoch + 1}/{args.epochs} | train loss: {train_recon_loss:.5f} | val loss: {val_loss:.5f}")

        # Best-checkpoint selection stays based on reconstruction val loss --
        # that's the metric that actually matters for the downstream encoder
        # weights, LOCA is only there to shape the representation, not to be
        # optimized for on its own.
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            ckpt = {"model_state_dict": model.state_dict(), "epoch": epoch, "val_loss": val_loss}
            if position_head is not None:
                ckpt["position_head_state_dict"] = position_head.state_dict()
            torch.save(ckpt, out_dir / args.run_name)
        if epoch % 5 == 0:
            visualize_reconstruction(
                masked_sar[args.visualization_tile],
                target_sar[args.visualization_tile],
                pred[args.visualization_tile],
                loss_mask[args.visualization_tile],
                f"training/visuals/reconstruction_epoch_{epoch:03d}.png",
            )

    print(f"Done. Best val loss: {best_val_loss:.5f}. Checkpoint: {out_dir}/{args.run_name}")


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="MAE-style pretraining for UAVSAR U-Net")
    parser.add_argument("--train_csv", type=Path, default="training/pretrain_csvs_new/train.csv")
    parser.add_argument("--val_csv", type=Path, default="training/pretrain_csvs_new/val.csv")
    parser.add_argument("--out_dir", type=Path, default="training/pretrain_weights")
    parser.add_argument("--patch_size", type=int, default=4)
    parser.add_argument("--mask_ratio", type=float, default=0.5)
    parser.add_argument("--base_channels", type=int, default=32)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--loss_fn", type=str, default="ssim")
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--num_workers", type=int, default=2)
    parser.add_argument("--run_name", type=str, default="ssim_0.5_4_twenty_scene_loca")
    parser.add_argument("--visualization_tile", type=int, default=3)

    # --- LOCA (position prediction auxiliary task) ---
    parser.add_argument(
        "--use_loca", action="store_true",
        help="Add a jigsaw-style position-prediction auxiliary loss alongside MAE reconstruction.",
    )
    parser.add_argument(
        "--loca_block_size", type=int, default=64,
        help="Side length of blocks that get spatially shuffled for the LOCA task. Must evenly divide tile H/W.",
    )
    parser.add_argument(
        "--loca_min_valid_frac", type=float, default=0.9,
        help="Minimum fraction of valid SAR pixels a block needs to be eligible for LOCA shuffling.",
    )
    parser.add_argument(
        "--lambda_pos", type=float, default=0.3,
        help="Weight of the LOCA position-prediction loss relative to reconstruction loss.",
    )
    parser.add_argument(
        "--lambda_pos_warmup_epochs", type=int, default=5,
        help="Linearly ramp lambda_pos from 0 to its full value over this many epochs.",
    )
    parser.add_argument(
        "--diagnose_grad_norms", action="store_true",
        help="Periodically print encoder gradient-norm contribution from recon_loss vs. "
             "weighted pos_loss, to check whether lambda_pos is actually large enough to matter.",
    )
    parser.add_argument(
        "--diagnose_every", type=int, default=50,
        help="Run the gradient-norm diagnostic every N training batches (only used with --diagnose_grad_norms).",
    )
    return parser


if __name__ == "__main__":
    args = build_arg_parser().parse_args()
    train(args)