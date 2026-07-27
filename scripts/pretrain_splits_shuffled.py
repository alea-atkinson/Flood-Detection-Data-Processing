#!/usr/bin/env python3
"""Scan a folder of UAVSAR tiles (nested by scene/flight-line) and produce
train/validation CSVs for MAE pretraining.

Tiles are pooled across ALL scenes, shuffled, and split by individual tile
to hit the requested val fraction exactly. Scenes are no longer kept intact
across the split -- tiles from the same scene can land in both train and
val. Note: if tiles within a scene overlap spatially or are otherwise
correlated, this can leak near-duplicate content between train and val.

Output CSVs have a single column, `uavsar_path`, matching what
MAEFloodTileDataset expects.

Example:
    python3 build_pretrain_splits.py \
        --data-root /path/to/uavsar_tiles \
        --train-csv train.csv \
        --val-csv val.csv \
        --val-fraction 0.2 \
        --group-depth 1 \
        --seed 42

--group-depth controls how many path components under --data-root define a
"scene" for grouping (used only for the summary printout). With the default
of 1:
    data_root/scene_A/tile_001.tif   -> scene "scene_A"
    data_root/scene_A/sub/tile_002.tif -> scene "scene_A"
    data_root/scene_B/tile_003.tif   -> scene "scene_B"
Increase it if your flight-line folders are nested deeper than one level.
"""

from __future__ import annotations

import argparse
import csv
import random
from pathlib import Path


def find_tiles(data_root: Path, extensions: set[str]) -> list[Path]:
    tiles = [
        p for p in data_root.rglob("*")
        if p.is_file() and p.suffix.lower() in extensions
    ]
    if not tiles:
        raise ValueError(f"No files with extensions {sorted(extensions)} found under {data_root}")
    return tiles


def group_by_scene(tiles: list[Path], data_root: Path, group_depth: int) -> dict[str, list[Path]]:
    groups: dict[str, list[Path]] = {}
    for tile in tiles:
        rel_parts = tile.relative_to(data_root).parts
        if len(rel_parts) <= group_depth:
            raise ValueError(
                f"{tile} is not nested at least {group_depth + 1} levels deep under {data_root}; "
                "reduce --group-depth or check --data-root."
            )
        scene = "/".join(rel_parts[:group_depth])
        groups.setdefault(scene, []).append(tile)
    return groups


def split_tiles(
    tiles: list[Path], val_fraction: float, seed: int
) -> tuple[list[Path], list[Path]]:
    """Shuffle all tiles together and split by individual tile, ignoring
    which scene each tile came from."""
    shuffled = list(tiles)
    random.Random(seed).shuffle(shuffled)

    target_val = round(len(shuffled) * val_fraction)
    val_tiles = shuffled[:target_val]
    train_tiles = shuffled[target_val:]
    return train_tiles, val_tiles


def write_csv(csv_path: Path, tiles: list[Path], data_root: Path) -> None:
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["uavsar_path"])
        for tile in tiles:
            path_str = "tiles/" + str(tile.relative_to(data_root))
            writer.writerow([path_str])


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build train/val CSV splits from a nested UAVSAR tile folder.")
    parser.add_argument("--data-root", type=Path, default="tiles", help="Root folder containing scene/flight-line subfolders of tiles.")
    parser.add_argument("--train-csv", type=Path, default="training/pretrain_csvs_shuffled/train.csv")
    parser.add_argument("--val-csv", type=Path, default="training/pretrain_csvs_shuffled/val.csv")
    parser.add_argument("--val-fraction", type=float, default=0.2)
    parser.add_argument("--group-depth", type=int, default=1, help="Path components under --data-root that define a scene (used only for the summary printout).")
    parser.add_argument("--extensions", nargs="+", default=[".tif", ".tiff"], help="File extensions to include (default: .tif .tiff).")
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    extensions = {e.lower() if e.startswith(".") else f".{e.lower()}" for e in args.extensions}

    tiles = find_tiles(args.data_root, extensions)
    groups = group_by_scene(tiles, args.data_root, args.group_depth)
    train_tiles, val_tiles = split_tiles(tiles, args.val_fraction, args.seed)

    write_csv(args.train_csv, train_tiles, args.data_root)
    write_csv(args.val_csv, val_tiles, args.data_root)

    total_tiles = len(train_tiles) + len(val_tiles)

    train_scene_set = {
        "/".join(t.relative_to(args.data_root).parts[:args.group_depth]) for t in train_tiles
    }
    val_scene_set = {
        "/".join(t.relative_to(args.data_root).parts[:args.group_depth]) for t in val_tiles
    }
    overlap = train_scene_set & val_scene_set

    print(f"Scenes found: {len(groups)} | Total tiles: {total_tiles}")
    print(f"Train: {len(train_tiles)} tiles (spanning {len(train_scene_set)} scenes) -> {args.train_csv}")
    print(f"Val:   {len(val_tiles)} tiles (spanning {len(val_scene_set)} scenes) -> {args.val_csv}")
    print(f"Scenes appearing in both train and val: {len(overlap)}")
    print(f"Actual val fraction (by tile count): {len(val_tiles) / total_tiles:.3f}")


if __name__ == "__main__":
    main()