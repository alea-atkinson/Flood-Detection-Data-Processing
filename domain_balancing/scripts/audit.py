#!/usr/bin/env python3

"""
Audit LOFPO training distributions for domain/flood-balanced training.

Purpose:
For each held-out LOFPO fold, inspect imbalance across:
1. source flight path
2. flood-fraction bin
3. source flight path × flood-fraction bin

This script does NOT train models.

Outputs:
- domain_balancing/tables/lofo_training_per_tile_audit.csv
- domain_balancing/tables/lofo_training_distribution_by_source_fp.csv
- domain_balancing/tables/lofo_training_distribution_by_flood_bin.csv
- domain_balancing/tables/lofo_training_distribution_by_source_fp_and_flood_bin.csv
- domain_balancing/summaries/lofo_training_distribution_audit_summary.txt
"""

import csv
import re
from pathlib import Path
from collections import defaultdict

import numpy as np
import pandas as pd
from PIL import Image
import rasterio


REPO_ROOT = Path("/mnt/linuxlab/home/reuuzheng/Flood-Detection-Data-Processing")

LOFO_DIR = REPO_ROOT / "training" / "lofo_csvs"
OUT_DIR = REPO_ROOT / "domain_balancing"
TABLES_DIR = OUT_DIR / "tables"
SUMMARIES_DIR = OUT_DIR / "summaries"

TABLES_DIR.mkdir(parents=True, exist_ok=True)
SUMMARIES_DIR.mkdir(parents=True, exist_ok=True)

FOLDS = ["fp1", "fp2", "fp3", "fp4", "fp5", "fp6", "fp7"]

FLOOD_BINS = [
    (0.00, 0.01, "0-1%"),
    (0.01, 0.05, "1-5%"),
    (0.05, 0.10, "5-10%"),
    (0.10, 0.25, "10-25%"),
    (0.25, 0.50, "25-50%"),
    (0.50, 1.01, ">50%"),
]


def resolve_path(path_value: str) -> Path:
    """
    Resolve paths from CSV. Handles absolute paths and repo-relative paths.
    """
    path = Path(str(path_value))

    if path.is_absolute():
        return path

    return REPO_ROOT / path


def find_column(df: pd.DataFrame, candidates: list[str]) -> str:
    for candidate in candidates:
        if candidate in df.columns:
            return candidate

    raise KeyError(
        f"Could not find any of columns {candidates}. "
        f"Available columns: {df.columns.tolist()}"
    )


def extract_source_fp(row: pd.Series, sar_col: str, mask_col: str) -> str:
    """
    Try to find source flight path from explicit CSV columns first,
    then fall back to parsing file paths.
    """
    explicit_cols = [
        "source_fp",
        "source_flight_path",
        "flight_path",
        "fp",
        "fold",
    ]

    for col in explicit_cols:
        if col in row.index and pd.notna(row[col]):
            value = str(row[col]).lower()
            match = re.search(r"fp[1-7]", value)
            if match:
                return match.group(0)

            if value in ["1", "2", "3", "4", "5", "6", "7"]:
                return f"fp{value}"

    combined = f"{row[sar_col]} {row[mask_col]}".lower()

    patterns = [
        r"(?<![a-z0-9])fp([1-7])(?![a-z0-9])",
        r"flight[_\- ]?path[_\- ]?([1-7])",
        r"heldout[_\-]?fp([1-7])",
    ]

    for pattern in patterns:
        match = re.search(pattern, combined)
        if match:
            return f"fp{match.group(1)}"

    return "unknown"


def read_image(path: Path) -> np.ndarray:
    """
    Read image files robustly.

    Uses rasterio for .tif/.tiff GeoTIFF files and PIL for PNG/JPG.
    Returns:
      - H x W for single-band images
      - H x W x C for multi-band images
    """
    if not path.exists():
        raise FileNotFoundError(f"Missing image file: {path}")

    suffix = path.suffix.lower()

    if suffix in [".tif", ".tiff"]:
        with rasterio.open(path) as src:
            arr = src.read()  # C, H, W

        if arr.shape[0] == 1:
            return arr[0]  # H, W

        return np.transpose(arr, (1, 2, 0))  # H, W, C

    return np.array(Image.open(path))  


def valid_mask_from_sar(sar: np.ndarray) -> np.ndarray:
    """
    Use the same basic valid-pixel idea as training normalization:
    pixels where all SAR channels are zero are invalid.

    If shape is HxWxC, invalid = all channels zero.
    If shape is CxHxW, convert logic accordingly.
    If grayscale or unexpected, treat all pixels as valid.
    """
    if sar.ndim == 3:
        # H, W, C
        if sar.shape[-1] in [3, 4]:
            return np.any(sar[..., :3] != 0, axis=-1)

        # C, H, W
        if sar.shape[0] in [3, 4]:
            return np.any(sar[:3, ...] != 0, axis=0)

    if sar.ndim == 2:
        return np.ones_like(sar, dtype=bool)

    return np.ones(sar.shape[:2], dtype=bool)


def flood_fraction_from_files(sar_path: Path, mask_path: Path) -> dict:
    sar = read_image(sar_path)
    mask = read_image(mask_path)

    if mask.ndim == 3:
        mask = mask[..., 0]

    valid = valid_mask_from_sar(sar)

    if valid.shape != mask.shape:
        raise ValueError(
            f"Shape mismatch:\n"
            f"  SAR valid mask shape: {valid.shape}\n"
            f"  flood mask shape: {mask.shape}\n"
            f"  SAR path: {sar_path}\n"
            f"  mask path: {mask_path}"
        )

    valid_pixels = int(valid.sum())

    if valid_pixels == 0:
        return {
            "valid_pixels": 0,
            "flood_pixels": 0,
            "flood_fraction": 0.0,
        }

    # For Florence binary masks, flooded pixels are positive.
    # Do not treat 255 as nodata here because Florence masks use 255-like values for flood.
    flood = (mask > 0) & valid

    flood_pixels = int(flood.sum())
    flood_fraction = flood_pixels / valid_pixels

    return {
        "valid_pixels": valid_pixels,
        "flood_pixels": flood_pixels,
        "flood_fraction": flood_fraction,
    }


def assign_flood_bin(flood_fraction: float) -> str:
    for low, high, label in FLOOD_BINS:
        if low <= flood_fraction < high:
            return label

    if flood_fraction >= 1.0:
        return ">50%"

    return "unknown"


def audit_fold(heldout_fold: str) -> list[dict]:
    csv_path = LOFO_DIR / f"heldout_{heldout_fold}" / "train.csv"

    if not csv_path.exists():
        raise FileNotFoundError(f"Missing train CSV: {csv_path}")

    df = pd.read_csv(csv_path)

    sar_col = find_column(
        df,
        [
            "uavsar_path",
            "sar_path",
            "image_path",
            "image",
            "input_path",
        ],
    )

    mask_col = find_column(
        df,
        [
            "flood_mask_path",
            "mask_path",
            "label_path",
            "target_path",
            "flood_path",
        ],
    )

    rows = []

    for idx, row in df.iterrows():
        sar_path = resolve_path(row[sar_col])
        mask_path = resolve_path(row[mask_col])
        source_fp = extract_source_fp(row, sar_col, mask_col)

        stats = flood_fraction_from_files(sar_path, mask_path)
        flood_bin = assign_flood_bin(stats["flood_fraction"])

        rows.append(
            {
                "heldout_fold": heldout_fold,
                "row_index": idx,
                "source_fp": source_fp,
                "flood_bin": flood_bin,
                "flood_fraction": stats["flood_fraction"],
                "valid_pixels": stats["valid_pixels"],
                "flood_pixels": stats["flood_pixels"],
                "sar_path": str(sar_path),
                "flood_mask_path": str(mask_path),
            }
        )

    return rows


def summarize_group(df: pd.DataFrame, group_cols: list[str]) -> pd.DataFrame:
    grouped = (
        df.groupby(group_cols, dropna=False)
        .agg(
            n_tiles=("flood_fraction", "size"),
            mean_flood_fraction=("flood_fraction", "mean"),
            median_flood_fraction=("flood_fraction", "median"),
            min_flood_fraction=("flood_fraction", "min"),
            max_flood_fraction=("flood_fraction", "max"),
            total_valid_pixels=("valid_pixels", "sum"),
            total_flood_pixels=("flood_pixels", "sum"),
        )
        .reset_index()
    )

    grouped["total_flood_fraction"] = (
        grouped["total_flood_pixels"] / grouped["total_valid_pixels"].replace(0, np.nan)
    ).fillna(0.0)

    # Percent within each heldout fold
    total_by_fold = grouped.groupby("heldout_fold")["n_tiles"].transform("sum")
    grouped["percent_tiles_within_fold"] = grouped["n_tiles"] / total_by_fold

    return grouped


def write_audit_summary(
    per_tile: pd.DataFrame,
    by_source: pd.DataFrame,
    by_bin: pd.DataFrame,
    by_cross: pd.DataFrame,
) -> None:
    path = SUMMARIES_DIR / "lofo_training_distribution_audit_summary.txt"

    lines = []
    lines.append("LOFPO Training Distribution Audit")
    lines.append("=" * 40)
    lines.append("")

    lines.append(f"Total audited training rows: {len(per_tile)}")
    lines.append("")

    unknown_source = per_tile[per_tile["source_fp"] == "unknown"]
    lines.append(f"Rows with unknown source_fp: {len(unknown_source)}")
    if len(unknown_source) > 0:
        lines.append("WARNING: source flight path could not be extracted for some rows.")
        lines.append("Inspect per-tile audit CSV before using domain-balanced sampling.")
    lines.append("")

    lines.append("Training tile counts by held-out fold:")
    for fold, count in per_tile.groupby("heldout_fold").size().items():
        lines.append(f"  {fold}: {count}")

    lines.append("")
    lines.append("Source-flight-path imbalance by held-out fold:")
    for fold in FOLDS:
        sub = by_source[by_source["heldout_fold"] == fold]
        if sub.empty:
            continue
        min_count = int(sub["n_tiles"].min())
        max_count = int(sub["n_tiles"].max())
        ratio = max_count / max(min_count, 1)
        lines.append(f"  {fold}: min={min_count}, max={max_count}, max/min={ratio:.2f}")

    lines.append("")
    lines.append("Flood-bin imbalance by held-out fold:")
    for fold in FOLDS:
        sub = by_bin[by_bin["heldout_fold"] == fold]
        if sub.empty:
            continue
        min_count = int(sub["n_tiles"].min())
        max_count = int(sub["n_tiles"].max())
        ratio = max_count / max(min_count, 1)
        lines.append(f"  {fold}: min={min_count}, max={max_count}, max/min={ratio:.2f}")

    lines.append("")
    lines.append("Source_fp x flood_bin sparsity:")
    for fold in FOLDS:
        sub = by_cross[by_cross["heldout_fold"] == fold]
        if sub.empty:
            continue
        n_groups = len(sub)
        n_tiny = int((sub["n_tiles"] < 5).sum())
        lines.append(f"  {fold}: groups={n_groups}, groups_with_<5_tiles={n_tiny}")

    lines.append("")
    lines.append("Interpretation guide:")
    lines.append("- If source_fp max/min ratio is high, source-domain-balanced sampling may be justified.")
    lines.append("- If flood-bin max/min ratio is high, flood-fraction-balanced sampling may be justified.")
    lines.append("- If many source_fp x flood_bin groups have <5 tiles, combined balancing may be too sparse or may need smoothing/clipping.")
    lines.append("- Do not train until unknown source_fp rows are resolved.")

    path.write_text("\n".join(lines))
    print(f"Wrote {path}")


def main() -> None:
    all_rows = []

    for fold in FOLDS:
        print(f"Auditing training distribution for heldout_{fold}...")
        fold_rows = audit_fold(fold)
        all_rows.extend(fold_rows)

    per_tile = pd.DataFrame(all_rows)

    per_tile_out = TABLES_DIR / "lofo_training_per_tile_audit.csv"
    by_source_out = TABLES_DIR / "lofo_training_distribution_by_source_fp.csv"
    by_bin_out = TABLES_DIR / "lofo_training_distribution_by_flood_bin.csv"
    by_cross_out = TABLES_DIR / "lofo_training_distribution_by_source_fp_and_flood_bin.csv"

    by_source = summarize_group(per_tile, ["heldout_fold", "source_fp"])
    by_bin = summarize_group(per_tile, ["heldout_fold", "flood_bin"])
    by_cross = summarize_group(per_tile, ["heldout_fold", "source_fp", "flood_bin"])

    per_tile.to_csv(per_tile_out, index=False)
    by_source.to_csv(by_source_out, index=False)
    by_bin.to_csv(by_bin_out, index=False)
    by_cross.to_csv(by_cross_out, index=False)

    print(f"Wrote {per_tile_out}")
    print(f"Wrote {by_source_out}")
    print(f"Wrote {by_bin_out}")
    print(f"Wrote {by_cross_out}")

    write_audit_summary(per_tile, by_source, by_bin, by_cross)

    print()
    print("Done. First inspect:")
    print(f"  {SUMMARIES_DIR / 'lofo_training_distribution_audit_summary.txt'}")
    print(f"  {by_source_out}")
    print(f"  {by_bin_out}")
    print(f"  {by_cross_out}")


if __name__ == "__main__":
    main()