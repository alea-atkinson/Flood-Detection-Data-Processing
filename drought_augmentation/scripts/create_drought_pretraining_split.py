#!/usr/bin/env python3
"""
Create train/val/test CSVs for assumed-negative drought pretraining.

This is weakly supervised negative pretraining, not SSL.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--inventory-csv",
        type=Path,
        default=Path("drought_augmentation/tables/drought_combined_inventory_relative.csv"),
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path("drought_augmentation/csvs/drought_pretraining"),
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--train-frac", type=float, default=0.8)
    parser.add_argument("--val-frac", type=float, default=0.1)
    args = parser.parse_args()

    df = pd.read_csv(args.inventory_csv)

    required = {"uavsar_path", "flood_mask_path"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    df = df[["uavsar_path", "flood_mask_path"]].copy()
    df = df.sample(frac=1.0, random_state=args.seed).reset_index(drop=True)

    n = len(df)
    n_train = int(n * args.train_frac)
    n_val = int(n * args.val_frac)

    train = df.iloc[:n_train]
    val = df.iloc[n_train:n_train + n_val]
    test = df.iloc[n_train + n_val:]

    args.out_dir.mkdir(parents=True, exist_ok=True)

    train.to_csv(args.out_dir / "train.csv", index=False)
    val.to_csv(args.out_dir / "val.csv", index=False)
    test.to_csv(args.out_dir / "test.csv", index=False)

    print(f"Input rows: {n}")
    print(f"Train: {len(train)} -> {args.out_dir / 'train.csv'}")
    print(f"Val:   {len(val)} -> {args.out_dir / 'val.csv'}")
    print(f"Test:  {len(test)} -> {args.out_dir / 'test.csv'}")


if __name__ == "__main__":
    main()