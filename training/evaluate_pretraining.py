#!/usr/bin/env python3
"""
Standalone reconstruction evaluator for UAVSAR masked-reconstruction pretraining.

This script:
1. Loads the pretrained U-Net checkpoint.
2. Reads validation UAVSAR tiles from a CSV with a `uavsar_path` column.
3. Applies the SAME normalization used in training/pretrain.py.
4. Generates a deterministic patch mask for each tile using:
      global seed + stable hash(tile path)
5. Computes masked-region reconstruction metrics:
      - overall MSE
      - overall MAE
      - per-band MSE
      - per-band MAE
6. Compares the trained model against two trivial baselines:
      - zero fill
      - visible-mean fill
7. Saves:
      - reconstruction_metrics.csv
      - per_scene_metrics.csv
      - summary.txt
      - example visualization PNGs

Run from the repo root:

    python3 training/evaluate_pretraining.py

Example with explicit paths:

    python3 training/evaluate_pretraining.py \
        --val-csv training/pretrain_csvs/val.csv \
        --checkpoint training/pretrain_weights/best_mae.pth \
        --out-dir training/pretrain_evaluation \
        --seed 42 \
        --num-examples 12

Notes:
- The evaluator intentionally does NOT reuse the stochastic masking function from
  pretrain.py because validation masks must remain fixed across repeated runs.
- The U-Net architecture and normalization behavior are imported from pretrain.py
  so the evaluator stays aligned with the training implementation.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import rasterio
import torch

from pretrain import MAEFloodTileDataset, UNet


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate UAVSAR masked-reconstruction pretraining."
    )
    parser.add_argument(
        "--val-csv",
        type=Path,
        default=Path("training/pretrain_csvs/val.csv"),
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=Path("training/pretrain_weights/best_mae.pth"),
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path("training/pretrain_evaluation"),
    )
    parser.add_argument("--patch-size", type=int, default=16)
    parser.add_argument("--mask-ratio", type=float, default=0.5)
    parser.add_argument("--min-valid-frac", type=float, default=0.9)
    parser.add_argument("--base-channels", type=int, default=32)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--num-examples",
        type=int,
        default=12,
        help="Number of deterministic example tiles to visualize.",
    )
    parser.add_argument(
        "--max-tiles",
        type=int,
        default=None,
        help="Optional limit for quick smoke tests. Default: evaluate all validation tiles.",
    )
    parser.add_argument(
        "--device",
        default="cuda" if torch.cuda.is_available() else "cpu",
    )
    return parser.parse_args()


def read_csv_rows(csv_path: Path) -> list[dict[str, str]]:
    with csv_path.open("r", newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        if "uavsar_path" not in (reader.fieldnames or []):
            raise ValueError(f"{csv_path} is missing required column 'uavsar_path'")
        return list(reader)


def stable_tile_seed(tile_path: str, global_seed: int) -> int:
    payload = f"{global_seed}:{tile_path}".encode("utf-8")
    digest = hashlib.sha256(payload).digest()
    return int.from_bytes(digest[:4], byteorder="little", signed=False)


def generate_deterministic_patch_mask(
    valid: np.ndarray,
    patch_size: int,
    mask_ratio: float,
    min_valid_frac: float,
    seed: int,
) -> np.ndarray:
    if valid.ndim != 2:
        raise ValueError(f"`valid` must be 2-D, got shape {valid.shape}")

    h, w = valid.shape
    if h % patch_size != 0 or w % patch_size != 0:
        raise ValueError(
            f"Tile shape ({h}, {w}) is not divisible by patch_size={patch_size}"
        )

    n_rows = h // patch_size
    n_cols = w // patch_size
    eligible: list[tuple[int, int]] = []

    for r in range(n_rows):
        for c in range(n_cols):
            block = valid[
                r * patch_size : (r + 1) * patch_size,
                c * patch_size : (c + 1) * patch_size,
            ]
            if float(block.mean()) >= min_valid_frac:
                eligible.append((r, c))

    num_to_mask = int(len(eligible) * mask_ratio)
    rng = np.random.default_rng(seed)
    order = rng.permutation(len(eligible))

    patch_mask = np.zeros((n_rows, n_cols), dtype=bool)

    for idx in order[:num_to_mask]:
        r, c = eligible[int(idx)]
        patch_mask[r, c] = True

    return np.kron(
        patch_mask,
        np.ones((patch_size, patch_size), dtype=bool),
    )


def load_and_prepare_tile(
    tile_path: str,
    patch_size: int,
    mask_ratio: float,
    min_valid_frac: float,
    seed: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    with rasterio.open(tile_path) as src:
        sar = src.read().astype(np.float32)

    if sar.shape[0] < 3:
        raise ValueError(
            f"Expected at least 3 SAR bands, got {sar.shape[0]} in {tile_path}"
        )

    sar = sar[:3]

    valid = ~(sar == 0).all(axis=0)
    sar[:, ~valid] = np.nan

    sar = MAEFloodTileDataset._normalize_per_tile(sar, valid)

    h, w = valid.shape
    h_crop = (h // patch_size) * patch_size
    w_crop = (w // patch_size) * patch_size

    sar = sar[:, :h_crop, :w_crop]
    valid = valid[:h_crop, :w_crop]

    tile_seed = stable_tile_seed(tile_path, seed)

    full_mask = generate_deterministic_patch_mask(
        valid=valid,
        patch_size=patch_size,
        mask_ratio=mask_ratio,
        min_valid_frac=min_valid_frac,
        seed=tile_seed,
    )

    masked_sar = sar.copy()
    masked_sar[:, full_mask] = 0.0

    loss_mask = (full_mask & valid).astype(np.float32)[None, :, :]

    return masked_sar, sar, loss_mask, valid


def compute_masked_metrics(
    pred: torch.Tensor,
    target: torch.Tensor,
    loss_mask: torch.Tensor,
) -> dict[str, float]:
    if pred.shape != target.shape:
        raise ValueError(f"Shape mismatch: pred={pred.shape}, target={target.shape}")

    if loss_mask.sum().item() < 1:
        raise ValueError("Loss mask contains no masked valid pixels.")

    mask = loss_mask.bool().expand_as(pred)
    error = pred - target

    result: dict[str, float] = {
        "mse": (error[mask] ** 2).mean().item(),
        "mae": error[mask].abs().mean().item(),
    }

    for band in range(pred.shape[1]):
        band_mask = loss_mask[:, 0].bool()
        band_error = error[:, band]
        result[f"band{band + 1}_mse"] = (
            band_error[band_mask] ** 2
        ).mean().item()
        result[f"band{band + 1}_mae"] = (
            band_error[band_mask].abs()
        ).mean().item()

    return result


def build_zero_fill_baseline(masked_input: torch.Tensor) -> torch.Tensor:
    return masked_input.clone()


def build_visible_mean_fill_baseline(
    masked_input: torch.Tensor,
    valid_mask: torch.Tensor,
    loss_mask: torch.Tensor,
) -> torch.Tensor:
    baseline = masked_input.clone()
    visible_valid = valid_mask.bool() & (~loss_mask.bool())

    for band in range(masked_input.shape[1]):
        band_values = masked_input[:, band : band + 1]
        values = band_values[visible_valid]

        fill_value = (
            values.mean()
            if values.numel() > 0
            else torch.tensor(0.0, device=masked_input.device)
        )

        band_tensor = baseline[:, band : band + 1]
        band_tensor[loss_mask.bool()] = fill_value

    return baseline


def display_rgb(img_chw: np.ndarray) -> np.ndarray:
    img = np.moveaxis(img_chw, 0, -1).astype(np.float32)

    lo = np.nanpercentile(img, 1.0)
    hi = np.nanpercentile(img, 99.0)

    if hi <= lo:
        return np.zeros_like(img)

    img = np.clip(img, lo, hi)
    img = (img - lo) / (hi - lo + 1e-8)
    return img


def save_visualization(
    tile_path: str,
    masked_input: torch.Tensor,
    target: torch.Tensor,
    prediction: torch.Tensor,
    loss_mask: torch.Tensor,
    save_path: Path,
) -> None:
    inp = masked_input[0].detach().cpu().numpy()
    tgt = target[0].detach().cpu().numpy()
    pred = prediction[0].detach().cpu().numpy()
    mask = loss_mask[0, 0].detach().cpu().numpy()

    abs_error = np.mean(np.abs(pred - tgt), axis=0)

    fig, axes = plt.subplots(1, 5, figsize=(22, 5))

    axes[0].imshow(display_rgb(tgt))
    axes[0].set_title("Original")

    axes[1].imshow(display_rgb(inp))
    axes[1].set_title("Masked Input")

    axes[2].imshow(display_rgb(pred))
    axes[2].set_title("Reconstruction")

    axes[3].imshow(abs_error, cmap="gray")
    axes[3].set_title("Absolute Error")

    axes[4].imshow(mask, cmap="gray", vmin=0, vmax=1)
    axes[4].set_title("Masked Valid Pixels")

    for ax in axes:
        ax.axis("off")

    fig.suptitle(tile_path, fontsize=10)
    plt.tight_layout()

    save_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def infer_scene(tile_path: str) -> str:
    return Path(tile_path).parent.name


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)

    if not rows:
        raise ValueError(f"No rows to write: {path}")

    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def aggregate_mean(rows: list[dict[str, object]], field: str) -> float:
    values = [float(row[field]) for row in rows]
    return float(np.mean(values))


def build_scene_summary(
    metric_rows: list[dict[str, object]],
) -> list[dict[str, object]]:
    grouped: dict[str, list[dict[str, object]]] = defaultdict(list)

    for row in metric_rows:
        grouped[str(row["scene"])].append(row)

    summary_rows: list[dict[str, object]] = []

    for scene in sorted(grouped):
        rows = grouped[scene]

        summary_rows.append(
            {
                "scene": scene,
                "tiles": len(rows),
                "model_mse": aggregate_mean(rows, "model_mse"),
                "model_mae": aggregate_mean(rows, "model_mae"),
                "zero_mse": aggregate_mean(rows, "zero_mse"),
                "zero_mae": aggregate_mean(rows, "zero_mae"),
                "mean_fill_mse": aggregate_mean(rows, "mean_fill_mse"),
                "mean_fill_mae": aggregate_mean(rows, "mean_fill_mae"),
                "effective_mask_fraction": aggregate_mean(
                    rows, "effective_mask_fraction"
                ),
            }
        )

    return summary_rows


def main() -> None:
    args = parse_args()

    if not args.val_csv.exists():
        raise FileNotFoundError(f"Validation CSV not found: {args.val_csv}")

    if not args.checkpoint.exists():
        raise FileNotFoundError(f"Checkpoint not found: {args.checkpoint}")

    device = torch.device(args.device)
    print(f"Using device: {device}")

    rows = read_csv_rows(args.val_csv)

    if args.max_tiles is not None:
        rows = rows[: args.max_tiles]

    if not rows:
        raise ValueError("Validation CSV contains no rows.")

    print(f"Validation tiles to evaluate: {len(rows)}")

    model = UNet(
        in_channels=3,
        out_channels=3,
        base_channels=args.base_channels,
    ).to(device)

    checkpoint = torch.load(
        args.checkpoint,
        map_location=device,
        weights_only=False,
    )

    state = (
        checkpoint["model_state_dict"]
        if "model_state_dict" in checkpoint
        else checkpoint
    )

    model.load_state_dict(state)
    model.eval()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    examples_dir = args.out_dir / "examples"
    examples_dir.mkdir(parents=True, exist_ok=True)

    metric_rows: list[dict[str, object]] = []

    num_examples = min(args.num_examples, len(rows))
    if num_examples > 0:
        example_indices = set(
            np.linspace(0, len(rows) - 1, num_examples, dtype=int).tolist()
        )
    else:
        example_indices = set()

    with torch.no_grad():
        for idx, row in enumerate(rows):
            tile_path = row["uavsar_path"]
            scene = infer_scene(tile_path)

            masked_np, target_np, loss_mask_np, valid_np = load_and_prepare_tile(
                tile_path=tile_path,
                patch_size=args.patch_size,
                mask_ratio=args.mask_ratio,
                min_valid_frac=args.min_valid_frac,
                seed=args.seed,
            )

            masked = torch.from_numpy(masked_np)[None].to(device)
            target = torch.from_numpy(target_np)[None].to(device)
            loss_mask = torch.from_numpy(loss_mask_np)[None].to(device)
            valid_mask = (
                torch.from_numpy(valid_np.astype(np.float32))[None, None].to(device)
            )

            if loss_mask.sum().item() < 1:
                print(f"Skipping tile with no masked valid pixels: {tile_path}")
                continue

            pred = model(masked)

            model_metrics = compute_masked_metrics(pred, target, loss_mask)

            zero_pred = build_zero_fill_baseline(masked)
            zero_metrics = compute_masked_metrics(
                zero_pred, target, loss_mask
            )

            mean_pred = build_visible_mean_fill_baseline(
                masked,
                valid_mask,
                loss_mask,
            )
            mean_metrics = compute_masked_metrics(
                mean_pred, target, loss_mask
            )

            masked_valid_pixels = int(loss_mask.sum().item())
            valid_pixels = int(valid_mask.sum().item())
            effective_mask_fraction = (
                masked_valid_pixels / valid_pixels
                if valid_pixels > 0
                else 0.0
            )

            metric_rows.append(
                {
                    "tile_path": tile_path,
                    "scene": scene,
                    "masked_valid_pixels": masked_valid_pixels,
                    "valid_pixels": valid_pixels,
                    "effective_mask_fraction": effective_mask_fraction,
                    "model_mse": model_metrics["mse"],
                    "model_mae": model_metrics["mae"],
                    "model_band1_mse": model_metrics["band1_mse"],
                    "model_band2_mse": model_metrics["band2_mse"],
                    "model_band3_mse": model_metrics["band3_mse"],
                    "model_band1_mae": model_metrics["band1_mae"],
                    "model_band2_mae": model_metrics["band2_mae"],
                    "model_band3_mae": model_metrics["band3_mae"],
                    "zero_mse": zero_metrics["mse"],
                    "zero_mae": zero_metrics["mae"],
                    "mean_fill_mse": mean_metrics["mse"],
                    "mean_fill_mae": mean_metrics["mae"],
                }
            )

            if idx in example_indices:
                safe_scene = scene.replace("/", "_")
                save_path = examples_dir / (
                    f"{idx:04d}_{safe_scene}_{Path(tile_path).stem}.png"
                )

                save_visualization(
                    tile_path=tile_path,
                    masked_input=masked,
                    target=target,
                    prediction=pred,
                    loss_mask=loss_mask,
                    save_path=save_path,
                )

            if (idx + 1) % 25 == 0 or idx + 1 == len(rows):
                print(f"Evaluated {idx + 1}/{len(rows)} tiles")

    if not metric_rows:
        raise RuntimeError("No tiles were successfully evaluated.")

    metrics_path = args.out_dir / "reconstruction_metrics.csv"
    write_csv(metrics_path, metric_rows)

    scene_rows = build_scene_summary(metric_rows)
    scene_path = args.out_dir / "per_scene_metrics.csv"
    write_csv(scene_path, scene_rows)

    summary = {
        "tiles_evaluated": len(metric_rows),
        "model_mse": aggregate_mean(metric_rows, "model_mse"),
        "model_mae": aggregate_mean(metric_rows, "model_mae"),
        "zero_mse": aggregate_mean(metric_rows, "zero_mse"),
        "zero_mae": aggregate_mean(metric_rows, "zero_mae"),
        "mean_fill_mse": aggregate_mean(metric_rows, "mean_fill_mse"),
        "mean_fill_mae": aggregate_mean(metric_rows, "mean_fill_mae"),
        "effective_mask_fraction": aggregate_mean(
            metric_rows, "effective_mask_fraction"
        ),
    }

    summary_lines = [
        "UAVSAR Pretraining Reconstruction Evaluation",
        "===========================================",
        f"Checkpoint: {args.checkpoint}",
        f"Validation CSV: {args.val_csv}",
        f"Tiles evaluated: {summary['tiles_evaluated']}",
        f"Patch size: {args.patch_size}",
        f"Mask ratio among eligible patches: {args.mask_ratio}",
        f"Minimum valid fraction per eligible patch: {args.min_valid_frac}",
        f"Deterministic mask seed: {args.seed}",
        "",
        "Overall masked-region metrics",
        "-----------------------------",
        f"Model MSE:       {summary['model_mse']:.6f}",
        f"Model MAE:       {summary['model_mae']:.6f}",
        f"Zero-fill MSE:   {summary['zero_mse']:.6f}",
        f"Zero-fill MAE:   {summary['zero_mae']:.6f}",
        f"Mean-fill MSE:   {summary['mean_fill_mse']:.6f}",
        f"Mean-fill MAE:   {summary['mean_fill_mae']:.6f}",
        "",
        (
            "Mean effective mask fraction of valid pixels: "
            f"{summary['effective_mask_fraction']:.4f}"
        ),
        "",
        f"Per-tile metrics: {metrics_path}",
        f"Per-scene metrics: {scene_path}",
        f"Examples: {examples_dir}",
    ]

    summary_path = args.out_dir / "summary.txt"
    summary_path.write_text("\n".join(summary_lines) + "\n", encoding="utf-8")

    print()
    print("\n".join(summary_lines))


if __name__ == "__main__":
    main()
