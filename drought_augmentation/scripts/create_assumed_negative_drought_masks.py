#!/usr/bin/env python3
"""
Create all-zero assumed-negative masks for drought UAVSAR tiles.

This script:
- reads drought SAR tiles
- creates one single-band uint8 mask per tile
- writes zeros everywhere the tile footprint exists
- copies geospatial profile from the SAR tile
- optionally skips tiles below a valid-pixel coverage threshold
- writes an inventory CSV

Important:
These are assumed-negative weak labels, not verified no-flood labels.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import rasterio


def valid_pixel_fraction(sar_array: np.ndarray) -> float:
    """
    Valid SAR pixels are pixels where at least one of the first 3 channels is nonzero.
    sar_array shape: bands, height, width
    """
    bands = min(3, sar_array.shape[0])
    valid = np.any(sar_array[:bands] != 0, axis=0)
    return float(valid.mean())


def create_zero_mask_for_tile(
    sar_path: Path,
    mask_path: Path,
    nodata_value: int | None,
) -> dict[str, Any]:
    with rasterio.open(sar_path) as src:
        sar = src.read()
        frac_valid = valid_pixel_fraction(sar)

        profile = src.profile.copy()
        profile.update(
            count=1,
            dtype="uint8",
            nodata=nodata_value,
            compress="lzw",
        )

        mask = np.zeros((src.height, src.width), dtype=np.uint8)

    mask_path.parent.mkdir(parents=True, exist_ok=True)

    with rasterio.open(mask_path, "w", **profile) as dst:
        dst.write(mask, 1)

    return {
        "uavsar_path": str(sar_path),
        "flood_mask_path": str(mask_path),
        "valid_pixel_fraction": frac_valid,
        "assumed_flood_fraction": 0.0,
        "label_type": "assumed_negative_drought",
        "height": int(mask.shape[0]),
        "width": int(mask.shape[1]),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--tile-dir",
        type=Path,
        required=True,
        help="Directory containing drought SAR tiles.",
    )
    parser.add_argument(
        "--mask-dir",
        type=Path,
        required=True,
        help="Output directory for generated all-zero masks.",
    )
    parser.add_argument(
        "--inventory-csv",
        type=Path,
        required=True,
        help="Output inventory CSV path.",
    )
    parser.add_argument(
        "--min-valid-fraction",
        type=float,
        default=0.5,
        help="Skip SAR tiles below this valid-pixel fraction. Default: 0.5.",
    )
    parser.add_argument(
        "--nodata-value",
        type=int,
        default=None,
        help="Optional nodata value for output masks. Default: None.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing masks.",
    )
    args = parser.parse_args()

    tile_dir = args.tile_dir
    mask_dir = args.mask_dir
    inventory_csv = args.inventory_csv

    if not tile_dir.exists():
        raise FileNotFoundError(f"Tile directory does not exist: {tile_dir}")

    sar_tiles = sorted(list(tile_dir.glob("*.tif")) + list(tile_dir.glob("*.tiff")))
    if not sar_tiles:
        raise FileNotFoundError(f"No .tif/.tiff tiles found in: {tile_dir}")

    records: list[dict[str, Any]] = []
    skipped: list[dict[str, Any]] = []

    for sar_path in sar_tiles:
        with rasterio.open(sar_path) as src:
            sar = src.read()
            frac_valid = valid_pixel_fraction(sar)

        if frac_valid < args.min_valid_fraction:
            skipped.append(
                {
                    "uavsar_path": str(sar_path),
                    "reason": "valid_fraction_below_threshold",
                    "valid_pixel_fraction": frac_valid,
                }
            )
            continue

        mask_path = mask_dir / sar_path.name

        if mask_path.exists() and not args.overwrite:
            raise FileExistsError(
                f"Mask already exists: {mask_path}. Use --overwrite to replace."
            )

        record = create_zero_mask_for_tile(
            sar_path=sar_path,
            mask_path=mask_path,
            nodata_value=args.nodata_value,
        )
        records.append(record)

    inventory_csv.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(records).to_csv(inventory_csv, index=False)

    skipped_csv = inventory_csv.with_name(inventory_csv.stem + "_skipped.csv")
    pd.DataFrame(skipped).to_csv(skipped_csv, index=False)

    print(f"Input SAR tiles: {len(sar_tiles)}")
    print(f"Masks created: {len(records)}")
    print(f"Skipped tiles: {len(skipped)}")
    print(f"Inventory CSV: {inventory_csv}")
    print(f"Skipped CSV: {skipped_csv}")
    print(f"Mask directory: {mask_dir}")


if __name__ == "__main__":
    main()