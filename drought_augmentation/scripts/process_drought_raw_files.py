#!/usr/bin/env python3
"""
Batch-process drought UAVSAR Pauli GeoTIFFs.

For each raw Pauli TIFF:
1. Copy from Downloads or another source folder into raw_files/
2. Run scripts/process_uavsar.py to produce filtered, 20m, and tiled outputs
3. Run create_assumed_negative_drought_masks.py to make all-zero masks
4. Write one inventory CSV per scene
5. Write a combined inventory CSV across all processed scenes

This script assumes these are drought/non-flood scenes.
The masks are assumed-negative weak labels, not verified no-flood labels.
"""

from __future__ import annotations

import argparse
import shutil
import subprocess
from pathlib import Path

import pandas as pd


def scene_stem_from_filename(path: Path) -> str:
    """
    Convert a raw UAVSAR filename into a clean drought scene name.

    Example:
    SMAPdr_27005_12117_000_121029_L090_CX_01_pauli.tif
    -> drought_SMAPdr_27005_12117
    """
    parts = path.stem.split("_")
    if len(parts) >= 3:
        return "drought_" + "_".join(parts[:3])
    return "drought_" + path.stem.replace("_pauli", "")


def run(cmd: list[str], dry_run: bool = False) -> None:
    print("\n$", " ".join(cmd))
    if not dry_run:
        subprocess.run(cmd, check=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=Path("/home/reuuzheng/Downloads"),
        help="Directory containing downloaded raw Pauli TIFFs.",
    )
    parser.add_argument(
        "--pattern",
        default="*pauli.tif",
        help="Glob pattern for raw files. Default: *pauli.tif",
    )
    parser.add_argument(
        "--repo-root",
        type=Path,
        default=Path("."),
        help="Repo root. Default: current directory.",
    )
    parser.add_argument(
        "--min-valid-fraction",
        type=float,
        default=0.5,
        help="Minimum valid SAR pixel fraction for keeping drought tiles.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing process_uavsar outputs and generated masks.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print commands without running them.",
    )
    args = parser.parse_args()

    repo_root = args.repo_root.resolve()
    input_dir = args.input_dir

    raw_dir = repo_root / "raw_files"
    raw_dir.mkdir(parents=True, exist_ok=True)

    tables_dir = repo_root / "drought_augmentation" / "tables"
    masks_root = repo_root / "drought_augmentation" / "generated_masks"
    tables_dir.mkdir(parents=True, exist_ok=True)
    masks_root.mkdir(parents=True, exist_ok=True)

    raw_sources = sorted(input_dir.glob(args.pattern))

    if not raw_sources:
        raise FileNotFoundError(
            f"No files matched {args.pattern!r} under {input_dir}"
        )

    print(f"Found {len(raw_sources)} raw TIFF(s).")
    for p in raw_sources:
        print(" ", p)

    inventory_paths: list[Path] = []

    for src in raw_sources:
        scene_name = scene_stem_from_filename(src)

        dst_raw = raw_dir / src.name
        tile_dir = repo_root / "tiles" / scene_name
        mask_dir = masks_root / scene_name
        inventory_csv = tables_dir / f"{scene_name}_inventory.csv"

        print("\n" + "=" * 80)
        print(f"Scene: {scene_name}")
        print(f"Source: {src}")
        print(f"Raw copy: {dst_raw}")
        print(f"Tile dir: {tile_dir}")
        print(f"Mask dir: {mask_dir}")
        print(f"Inventory: {inventory_csv}")

        if not dst_raw.exists() or args.overwrite:
            print(f"Copying raw TIFF to {dst_raw}")
            if not args.dry_run:
                shutil.copy2(src, dst_raw)
        else:
            print(f"Raw file already exists, skipping copy: {dst_raw}")

        process_cmd = [
            "python3",
            str(repo_root / "scripts" / "process_uavsar.py"),
            "--src",
            str(dst_raw),
            "--name",
            scene_name,
        ]
        if args.overwrite:
            process_cmd.append("--overwrite")

        run(process_cmd, dry_run=args.dry_run)

        mask_cmd = [
            "python3",
            str(repo_root / "drought_augmentation" / "scripts" / "create_assumed_negative_drought_masks.py"),
            "--tile-dir",
            str(tile_dir),
            "--mask-dir",
            str(mask_dir),
            "--inventory-csv",
            str(inventory_csv),
            "--min-valid-fraction",
            str(args.min_valid_fraction),
        ]
        if args.overwrite:
            mask_cmd.append("--overwrite")

        run(mask_cmd, dry_run=args.dry_run)

        inventory_paths.append(inventory_csv)

    if not args.dry_run:
        inventories = []
        for p in inventory_paths:
            df = pd.read_csv(p)
            df["scene_name"] = p.stem.replace("_inventory", "")
            inventories.append(df)

        combined = pd.concat(inventories, ignore_index=True)
        combined_csv = tables_dir / "drought_combined_inventory.csv"
        combined.to_csv(combined_csv, index=False)

        print("\n" + "=" * 80)
        print(f"Wrote combined inventory: {combined_csv}")
        print(f"Combined rows: {len(combined)}")


if __name__ == "__main__":
    main()