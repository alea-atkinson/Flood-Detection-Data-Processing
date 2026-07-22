#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import random
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable


@dataclass(frozen=True)
class Record:
    uavsar_path: Path
    flood_mask_path: Path

    @property
    def tile_name(self) -> str:
        return self.uavsar_path.name

    def as_csv_row(self, repo_root: Path) -> dict[str, str]:
        return {
            "uavsar_path": self.uavsar_path.relative_to(repo_root).as_posix(),
            "flood_mask_path": self.flood_mask_path.relative_to(repo_root).as_posix(),
        }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create strict LOFPO train/validation/test CSV splits."
    )
    parser.add_argument(
        "--dataset-root",
        type=Path,
        default=Path("2025_Tile_Data/Only_PNG_Data"),
    )
    parser.add_argument("--heldout-fp", type=str, default="fp1")
    parser.add_argument("--validation-fraction", type=float, default=0.20)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("training/lofo_csvs"),
    )
    return parser.parse_args()


def find_repo_root() -> Path:
    cwd = Path.cwd().resolve()
    if not (cwd / "training").exists() or not (cwd / "2025_Tile_Data").exists():
        raise RuntimeError(
            "Run this script from the repository root. "
            f"Current directory: {cwd}"
        )
    return cwd


def discover_flight_paths(dataset_root: Path) -> list[Path]:
    paths = [
        p for p in dataset_root.iterdir()
        if p.is_dir() and p.name.startswith("fp")
    ]

    def sort_key(p: Path):
        suffix = p.name.removeprefix("fp")
        return int(suffix) if suffix.isdigit() else p.name

    return sorted(paths, key=sort_key)


def build_records_for_flight_path(
    fp_dir: Path,
) -> tuple[list[Record], list[str], list[str]]:
    uavsar_dir = fp_dir / "UAVSAR"
    mask_dir = fp_dir / "flood_mask"

    if not uavsar_dir.is_dir():
        raise FileNotFoundError(f"Missing UAVSAR directory: {uavsar_dir}")
    if not mask_dir.is_dir():
        raise FileNotFoundError(f"Missing flood_mask directory: {mask_dir}")

    uavsar = {
        p.name: p.resolve()
        for p in sorted(uavsar_dir.glob("*.tif"))
        if p.is_file()
    }
    masks = {
        p.name: p.resolve()
        for p in sorted(mask_dir.glob("*.tif"))
        if p.is_file()
    }

    paired = sorted(set(uavsar) & set(masks))
    unmatched_uavsar = sorted(set(uavsar) - set(masks))
    unmatched_masks = sorted(set(masks) - set(uavsar))

    records = [
        Record(uavsar[name], masks[name])
        for name in paired
    ]
    return records, unmatched_uavsar, unmatched_masks


def group_by_tile_name(records: Iterable[Record]) -> dict[str, list[Record]]:
    grouped: dict[str, list[Record]] = defaultdict(list)
    for record in records:
        grouped[record.tile_name].append(record)
    return dict(grouped)


def split_groups(
    grouped: dict[str, list[Record]],
    validation_fraction: float,
    seed: int,
) -> tuple[list[Record], list[Record]]:
    names = sorted(grouped)
    rng = random.Random(seed)
    rng.shuffle(names)

    n_val = round(len(names) * validation_fraction)
    if len(names) >= 2:
        n_val = max(1, min(n_val, len(names) - 1))

    val_names = set(names[:n_val])
    train_names = set(names[n_val:])

    train = [
        record
        for name in sorted(train_names)
        for record in grouped[name]
    ]
    val = [
        record
        for name in sorted(val_names)
        for record in grouped[name]
    ]
    return train, val


def path_keys(records: Iterable[Record]) -> set[tuple[str, str]]:
    return {
        (r.uavsar_path.as_posix(), r.flood_mask_path.as_posix())
        for r in records
    }


def tile_names(records: Iterable[Record]) -> set[str]:
    return {r.tile_name for r in records}


def validate_splits(
    train: list[Record],
    val: list[Record],
    test: list[Record],
    heldout_fp: str,
) -> None:
    checks = {
        "train/validation full-path overlap":
            len(path_keys(train) & path_keys(val)),
        "train/test full-path overlap":
            len(path_keys(train) & path_keys(test)),
        "validation/test full-path overlap":
            len(path_keys(val) & path_keys(test)),
        "train/validation tile-filename overlap":
            len(tile_names(train) & tile_names(val)),
        "train/test tile-filename overlap":
            len(tile_names(train) & tile_names(test)),
        "validation/test tile-filename overlap":
            len(tile_names(val) & tile_names(test)),
    }

    print("\nOverlap validation:")
    for name, count in checks.items():
        print(f"  {name}: {count}")

    bad = {k: v for k, v in checks.items() if v != 0}
    if bad:
        raise RuntimeError(f"Invalid split; overlap detected: {bad}")

    for record in test:
        if heldout_fp not in record.uavsar_path.parts:
            raise RuntimeError(
                f"Test record not from held-out {heldout_fp}: "
                f"{record.uavsar_path}"
            )

    for split_name, records in (("train", train), ("validation", val)):
        bad_records = [
            r.uavsar_path
            for r in records
            if heldout_fp in r.uavsar_path.parts
        ]
        if bad_records:
            raise RuntimeError(
                f"{split_name} contains held-out {heldout_fp}. "
                f"Example: {bad_records[0]}"
            )

    print(f"  held-out flight-path membership check: passed ({heldout_fp})")


def write_csv(path: Path, records: list[Record], repo_root: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["uavsar_path", "flood_mask_path"],
        )
        writer.writeheader()
        for record in records:
            writer.writerow(record.as_csv_row(repo_root))


def main() -> None:
    args = parse_args()

    if not 0.0 < args.validation_fraction < 1.0:
        raise ValueError("--validation-fraction must be between 0 and 1.")

    repo_root = find_repo_root()
    dataset_root = (repo_root / args.dataset_root).resolve()
    output_root = (repo_root / args.output_root).resolve()

    if not dataset_root.is_dir():
        raise FileNotFoundError(f"Dataset root not found: {dataset_root}")

    fps = discover_flight_paths(dataset_root)
    if not fps:
        raise RuntimeError(f"No fp* folders found under {dataset_root}")

    fp_by_name = {p.name: p for p in fps}
    if args.heldout_fp not in fp_by_name:
        raise ValueError(
            f"{args.heldout_fp} not found. Available: {sorted(fp_by_name)}"
        )

    print("Discovered dataset structure:")
    for fp in fps:
        print(
            f"  {fp.name}: "
            f"UAVSAR={(fp/'UAVSAR').is_dir()} "
            f"flood_mask={(fp/'flood_mask').is_dir()} "
            f"land_cover={(fp/'land_cover').is_dir()}"
        )

    all_records: dict[str, list[Record]] = {}
    total_unmatched_uavsar = 0
    total_unmatched_masks = 0

    print("\nPairing summary:")
    for name, fp_dir in sorted(fp_by_name.items()):
        records, unmatched_uavsar, unmatched_masks = (
            build_records_for_flight_path(fp_dir)
        )
        all_records[name] = records
        total_unmatched_uavsar += len(unmatched_uavsar)
        total_unmatched_masks += len(unmatched_masks)

        print(
            f"  {name}: paired={len(records)} "
            f"unmatched_uavsar={len(unmatched_uavsar)} "
            f"unmatched_masks={len(unmatched_masks)}"
        )

    test = list(all_records[args.heldout_fp])
    test_names = tile_names(test)

    candidates: list[Record] = []
    for name, records in sorted(all_records.items()):
        if name != args.heldout_fp:
            candidates.extend(records)

    before_filter = len(candidates)
    filtered = [r for r in candidates if r.tile_name not in test_names]
    removed = before_filter - len(filtered)

    grouped = group_by_tile_name(filtered)
    train, val = split_groups(
        grouped,
        validation_fraction=args.validation_fraction,
        seed=args.seed,
    )

    validate_splits(train, val, test, args.heldout_fp)

    out_dir = output_root / f"heldout_{args.heldout_fp}"
    train_csv = out_dir / "train.csv"
    val_csv = out_dir / "validation.csv"
    test_csv = out_dir / "test.csv"

    write_csv(train_csv, train, repo_root)
    write_csv(val_csv, val, repo_root)
    write_csv(test_csv, test, repo_root)

    print("\nFinal split summary:")
    print(f"  held-out flight path: {args.heldout_fp}")
    print(f"  fp test records: {len(test)}")
    print(f"  candidate records before strict filtering: {before_filter}")
    print(f"  removed due to held-out filename overlap: {removed}")
    print(f"  train records: {len(train)}")
    print(f"  validation records: {len(val)}")
    print(f"  test records: {len(test)}")
    print(f"  unique train tile filenames: {len(tile_names(train))}")
    print(f"  unique validation tile filenames: {len(tile_names(val))}")
    print(f"  unique test tile filenames: {len(tile_names(test))}")
    print(f"  unmatched UAVSAR tiles skipped: {total_unmatched_uavsar}")
    print(f"  unmatched flood masks skipped: {total_unmatched_masks}")

    print("\nWrote:")
    print(f"  {train_csv.relative_to(repo_root)}")
    print(f"  {val_csv.relative_to(repo_root)}")
    print(f"  {test_csv.relative_to(repo_root)}")


if __name__ == "__main__":
    main()
