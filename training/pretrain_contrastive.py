#!/usr/bin/env python3
"""SimSiam-style contrastive pretraining for UAVSAR tiles.

This version is aligned with pretrain_mae.py:
  - uses training/pretrain_csvs/train.csv by default
  - uses training/pretrain_csvs/val.csv by default
  - expects a uavsar_path column
  - uses the same MAE/Fine-tune U-Net encoder naming:
        enc1, enc2, enc3, enc4, bottleneck, pool

Important:
  The checkpoint saves model_state_dict WITHOUT the final out.* layer.
  That prevents fine_tune.py from accidentally loading an untrained segmentation
  output head. Only encoder/bottleneck/decoder-compatible weights are transferred.
"""

from __future__ import annotations

import argparse
import csv
import importlib.util
import math
import random
from pathlib import Path
from typing import Any

import numpy as np
import rasterio
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


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


def read_uavsar_csv(csv_path: Path) -> list[Path]:
    rows: list[Path] = []
    with csv_path.open("r", newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        if "uavsar_path" not in (reader.fieldnames or []):
            raise ValueError(f"{csv_path} is missing required column 'uavsar_path'")
        for row in reader:
            rows.append(Path(row["uavsar_path"]))
    if not rows:
        raise ValueError(f"No rows found in {csv_path}")
    return rows


def resolve_path(path: Path, repo_root: Path) -> Path:
    if path.is_absolute():
        return path
    return repo_root / path


def load_uavsar_tensor(path: Path) -> torch.Tensor:
    """Load and normalize exactly in the spirit of pretrain_mae.py.

    Per-band normalization:
      - valid pixels are pixels where not all three SAR bands are zero
      - invalid pixels are set to zero
      - each band clips valid pixels to p1-p99
      - each band z-scores valid pixels
    """
    with rasterio.open(path) as src:
        sar = src.read().astype(np.float32)

    if sar.shape[0] < 3:
        raise ValueError(f"Expected at least 3 SAR bands, got {sar.shape[0]} in {path}")

    sar = sar[:3]
    valid = ~(sar == 0).all(axis=0)

    out = sar.copy()
    for c in range(out.shape[0]):
        band = out[c]
        values = band[valid]

        if values.size == 0:
            band[:] = 0.0
            out[c] = band
            continue

        low, high = np.percentile(values, [1.0, 99.0])
        values = np.clip(values, low, high)

        mean = values.mean()
        std = values.std()

        if std < 1e-6:
            band[:] = 0.0
        else:
            band[valid] = (values - mean) / std

        band[~valid] = 0.0
        out[c] = band

    return torch.from_numpy(out.astype(np.float32))


def sar_safe_augment(x: torch.Tensor) -> torch.Tensor:
    """SAR-safe augmentations.

    Avoid RGB natural-image transforms like hue shift, solarization, or strong
    color jitter. Use geometry and mild SAR-plausible perturbations.
    """
    y = x.clone()

    if random.random() < 0.5:
        y = torch.flip(y, dims=[2])  # horizontal
    if random.random() < 0.5:
        y = torch.flip(y, dims=[1])  # vertical

    k = random.randint(0, 3)
    if k:
        y = torch.rot90(y, k=k, dims=[1, 2])

    if random.random() < 0.8:
        y = y * random.uniform(0.85, 1.15)

    if random.random() < 0.5:
        y = y + torch.randn_like(y) * random.uniform(0.01, 0.05)

    if random.random() < 0.25:
        y = F.avg_pool2d(y.unsqueeze(0), kernel_size=3, stride=1, padding=1).squeeze(0)

    return y.contiguous()


class ContrastiveUAVSARDataset(Dataset):
    def __init__(self, paths: list[Path], repo_root: Path) -> None:
        self.paths = [resolve_path(path, repo_root) for path in paths]

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        x = load_uavsar_tensor(self.paths[index])
        return sar_safe_augment(x), sar_safe_augment(x)


class MAEStyleUNetEncoder(nn.Module):
    """Encoder wrapper for the U-Net architecture used in pretrain_mae.py."""

    def __init__(self, unet: nn.Module) -> None:
        super().__init__()
        self.unet = unet

        required = ["enc1", "enc2", "enc3", "enc4", "bottleneck", "pool"]
        missing = [name for name in required if not hasattr(unet, name)]
        if missing:
            raise AttributeError(
                "The imported UNet does not expose the expected MAE-style encoder "
                f"attributes {required}. Missing: {missing}."
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        enc1 = self.unet.enc1(x)
        enc2 = self.unet.enc2(self.unet.pool(enc1))
        enc3 = self.unet.enc3(self.unet.pool(enc2))
        enc4 = self.unet.enc4(self.unet.pool(enc3))
        bottleneck = self.unet.bottleneck(self.unet.pool(enc4))
        return bottleneck


class SimSiamPretrainer(nn.Module):
    def __init__(
        self,
        base_unet: nn.Module,
        image_size: int,
        projection_dim: int,
        hidden_dim: int,
    ) -> None:
        super().__init__()
        self.base_unet = base_unet
        self.encoder = MAEStyleUNetEncoder(base_unet)
        self.pool = nn.AdaptiveAvgPool2d((1, 1))

        with torch.no_grad():
            dummy = torch.zeros(1, 3, image_size, image_size)
            feature_dim = self.pool(self.encoder(dummy)).flatten(1).shape[1]

        self.projector = nn.Sequential(
            nn.Linear(feature_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, projection_dim),
            nn.BatchNorm1d(projection_dim, affine=False),
        )

        self.predictor = nn.Sequential(
            nn.Linear(projection_dim, hidden_dim // 2),
            nn.BatchNorm1d(hidden_dim // 2),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim // 2, projection_dim),
        )

    def encode_project_predict(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        features = self.pool(self.encoder(x)).flatten(1)
        z = self.projector(features)
        p = self.predictor(z)
        return p, z.detach()

    def forward(
        self,
        x1: torch.Tensor,
        x2: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        p1, z1 = self.encode_project_predict(x1)
        p2, z2 = self.encode_project_predict(x2)
        return p1, p2, z1, z2


def negative_cosine_similarity(p: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
    p = F.normalize(p, dim=1)
    z = F.normalize(z, dim=1)
    return -(p * z).sum(dim=1).mean()


def simsiam_loss(
    p1: torch.Tensor,
    p2: torch.Tensor,
    z1: torch.Tensor,
    z2: torch.Tensor,
) -> torch.Tensor:
    return 0.5 * (
        negative_cosine_similarity(p1, z2)
        + negative_cosine_similarity(p2, z1)
    )


def run_epoch(
    *,
    model: SimSiamPretrainer,
    loader: DataLoader,
    device: torch.device,
    optimizer: torch.optim.Optimizer | None,
) -> float:
    is_train = optimizer is not None
    model.train(is_train)

    total_loss = 0.0
    total_batches = 0

    for view1, view2 in loader:
        view1 = view1.to(device, non_blocking=True)
        view2 = view2.to(device, non_blocking=True)

        if is_train:
            optimizer.zero_grad(set_to_none=True)

        p1, p2, z1, z2 = model(view1, view2)
        loss = simsiam_loss(p1, p2, z1, z2)

        if is_train:
            loss.backward()
            optimizer.step()

        total_loss += float(loss.item())
        total_batches += 1

    if total_batches == 0:
        raise ValueError("No batches produced. Reduce batch size or increase sample count.")

    return total_loss / total_batches


def transferable_state_dict(model: nn.Module) -> dict[str, torch.Tensor]:
    """Save only weights that should transfer into segmentation.

    Excludes out.* because the SimSiam task never trains the segmentation head.
    """
    state = model.state_dict()
    return {key: value for key, value in state.items() if not key.startswith("out.")}


def write_metrics_header(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["epoch", "train_loss", "val_loss", "best_val_loss"])


def append_metrics(path: Path, epoch: int, train_loss: float, val_loss: float, best_val_loss: float) -> None:
    with path.open("a", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow([epoch, train_loss, val_loss, best_val_loss])


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="SimSiam-style contrastive pretraining for UAVSAR.")

    parser.add_argument("--repo-root", type=Path, default=Path("."))
    parser.add_argument("--train-csv", type=Path, default=Path("training/pretrain_csvs/train.csv"))
    parser.add_argument("--val-csv", type=Path, default=Path("training/pretrain_csvs/val.csv"))

    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--base-channels", type=int, default=32)
    parser.add_argument("--image-size", type=int, default=256)

    parser.add_argument("--projection-dim", type=int, default=256)
    parser.add_argument("--hidden-dim", type=int, default=512)

    parser.add_argument("--seed", type=int, default=98)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")

    parser.add_argument("--run-name", default="contrastive_siamese")
    parser.add_argument("--out-dir", type=Path, default=Path("training/pretrain_weights"))
    parser.add_argument("--results-dir", type=Path, default=Path("results"))

    parser.add_argument("--max-train-samples", type=int, default=None)
    parser.add_argument("--max-val-samples", type=int, default=None)

    return parser.parse_args()


def main() -> None:
    args = parse_args()
    seed_everything(args.seed)

    repo_root = args.repo_root.resolve()
    train_csv = (repo_root / args.train_csv).resolve()
    val_csv = (repo_root / args.val_csv).resolve()
    out_dir = (repo_root / args.out_dir).resolve()
    results_dir = (repo_root / args.results_dir).resolve()

    out_dir.mkdir(parents=True, exist_ok=True)
    results_dir.mkdir(parents=True, exist_ok=True)

    fine_tune = import_fine_tune_module(repo_root)

    train_paths = read_uavsar_csv(train_csv)
    val_paths = read_uavsar_csv(val_csv)

    if args.max_train_samples is not None:
        train_paths = train_paths[: args.max_train_samples]
    if args.max_val_samples is not None:
        val_paths = val_paths[: args.max_val_samples]

    if len(train_paths) < args.batch_size:
        raise ValueError(f"train samples {len(train_paths)} < batch size {args.batch_size}")
    if len(val_paths) < args.batch_size:
        raise ValueError(f"val samples {len(val_paths)} < batch size {args.batch_size}")

    device = torch.device(args.device)
    pin_memory = device.type == "cuda"

    train_loader = DataLoader(
        ContrastiveUAVSARDataset(train_paths, repo_root),
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=pin_memory,
        drop_last=True,
    )
    val_loader = DataLoader(
        ContrastiveUAVSARDataset(val_paths, repo_root),
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=pin_memory,
        drop_last=True,
    )

    # out_channels=1 matches segmentation architecture, but out.* is excluded
    # from saved checkpoint because the contrastive objective does not train it.
    base_unet = fine_tune.UNet(
        in_channels=3,
        out_channels=1,
        base_channels=args.base_channels,
    )

    model = SimSiamPretrainer(
        base_unet=base_unet,
        image_size=args.image_size,
        projection_dim=args.projection_dim,
        hidden_dim=args.hidden_dim,
    ).to(device)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )

    checkpoint_path = out_dir / f"best_{args.run_name}.pth"
    metrics_path = results_dir / f"{args.run_name}_pretrain_metrics.csv"
    write_metrics_header(metrics_path)

    print(f"Using device: {device}")
    print(f"Train CSV: {train_csv}")
    print(f"Val CSV: {val_csv}")
    print(f"Train samples: {len(train_paths)}")
    print(f"Val samples: {len(val_paths)}")
    print(f"Checkpoint path: {checkpoint_path}")
    print(f"Metrics path: {metrics_path}")

    best_val_loss = math.inf

    for epoch in range(1, args.epochs + 1):
        train_loss = run_epoch(
            model=model,
            loader=train_loader,
            device=device,
            optimizer=optimizer,
        )

        with torch.no_grad():
            val_loss = run_epoch(
                model=model,
                loader=val_loader,
                device=device,
                optimizer=None,
            )

        saved = ""
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(
                {
                    "model_state_dict": transferable_state_dict(model.base_unet),
                    "simsiam_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "epoch": epoch,
                    "val_loss": val_loss,
                    "best_val_loss": best_val_loss,
                    "pretrain_method": "simsiam_contrastive",
                    "args": {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
                },
                checkpoint_path,
            )
            saved = "saved"

        append_metrics(metrics_path, epoch, train_loss, val_loss, best_val_loss)

        print(
            f"Epoch {epoch:03d}/{args.epochs:03d} "
            f"train_loss={train_loss:.5f} "
            f"val_loss={val_loss:.5f} "
            f"best_val_loss={best_val_loss:.5f} {saved}"
        )

    print(f"Done. Best checkpoint: {checkpoint_path}")
    print(f"Metrics CSV: {metrics_path}")


if __name__ == "__main__":
    main()
