#!/usr/bin/env python3
"""
Create drought-augmented train CSVs from canonical no_overlap_converted splits.

This script:
- reads canonical base split CSVs from training/fine_tune_csvs/no_overlap_converted
- reads a drought inventory CSV created by create_assumed_negative_drought_masks.py
- appends drought assumed-negative rows to train CSVs only
- copies validation and test CSVs unchanged
- supports both:
    1. florence_only baseline
    2. with_milton baseline

It does NOT modify the original split files.
"""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path

import pandas as pd


def read_drought_rows(inventory_csv: Path) -> pd.DataFrame:
    inv = pd.read_csv(inventory_csv)

    required = {"uavsar_path", "flood_mask_path"}
    missing = required - set(inv.columns)
    if missing:
        raise ValueError(f"Inventory missing required columns: {missing}")

    drought = inv[["uavsar_path", "flood_mask_path"]].copy()
    drought["source_event"] = "drought"
    drought["label_type"] = "assumed_negative"

    return drought


def ensure_columns(base_df: pd.DataFrame, drought_df: pd.DataFrame) -> pd.DataFrame:
    """
    Make drought rows compatible with base CSV columns.

    If base CSV has only uavsar_path and flood_mask_path, output only those.
    If base CSV has extra columns, fill missing drought values with blanks.
    """
    out = drought_df.copy()

    for col in base_df.columns:
        if col not in out.columns:
            out[col] = ""

    return out[list(base_df.columns)]


def add_drought_to_train(
    base_train_csv: Path,
    output_train_csv: Path,
    drought_df: pd.DataFrame,
    drought_multiplier: int,
) -> dict:
    base = pd.read_csv(base_train_csv)
    drought_compatible = ensure_columns(base, drought_df)

    if drought_multiplier < 1:
        raise ValueError("--drought-multiplier must be >= 1")

    drought_repeated = pd.concat(
        [drought_compatible] * drought_multiplier,
        ignore_index=True,
    )

    augmented = pd.concat([base, drought_repeated], ignore_index=True)

    output_train_csv.parent.mkdir(parents=True, exist_ok=True)
    augmented.to_csv(output_train_csv, index=False)

    return {
        "base_train_csv": str(base_train_csv),
        "output_train_csv": str(output_train_csv),
        "base_rows": len(base),
        "drought_unique_rows": len(drought_compatible),
        "drought_multiplier": drought_multiplier,
        "drought_rows_added": len(drought_repeated),
        "output_rows": len(augmented),
    }


def copy_if_exists(src: Path, dst: Path) -> bool:
    if not src.exists():
        return False
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)
    return True


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--base-root",
        type=Path,
        default=Path("training/fine_tune_csvs/no_overlap_converted"),
        help="Canonical no_overlap_converted split root.",
    )
    parser.add_argument(
        "--condition",
        choices=["florence_only", "with_milton"],
        required=True,
        help="Which baseline split condition to augment.",
    )
    parser.add_argument(
        "--drought-inventory-csv",
        type=Path,
        required=True,
        help="Drought inventory CSV with uavsar_path and flood_mask_path.",
    )
    parser.add_argument(
        "--out-root",
        type=Path,
        required=True,
        help="Output root for augmented split CSVs.",
    )
    parser.add_argument(
        "--drought-multiplier",
        type=int,
        default=1,
        help="How many times to append the drought inventory to each train split. Default: 1.",
    )
    args = parser.parse_args()

    base_root = args.base_root
    condition = args.condition
    out_root = args.out_root

    if condition == "florence_only":
        train_root = base_root
        val_root = base_root
        test_root = base_root
    else:
        train_root = base_root / "with_milton"
        val_root = base_root / "with_milton"
        test_root = base_root

    drought_df = read_drought_rows(args.drought_inventory_csv)

    records = []

    for fp in range(1, 8):
        fold = f"fp{fp}"

        train_src = train_root / f"held_out_{fold}_train.csv"
        val_src = val_root / f"held_out_{fold}_val.csv"
        test_src = test_root / f"held_out_{fold}_test.csv"

        train_dst = out_root / f"held_out_{fold}_train.csv"
        val_dst = out_root / f"held_out_{fold}_val.csv"
        test_dst = out_root / f"held_out_{fold}_test.csv"

        if not train_src.exists():
            raise FileNotFoundError(f"Missing train CSV: {train_src}")
        if not val_src.exists():
            raise FileNotFoundError(f"Missing val CSV: {val_src}")
        if not test_src.exists():
            raise FileNotFoundError(f"Missing test CSV: {test_src}")

        rec = add_drought_to_train(
            base_train_csv=train_src,
            output_train_csv=train_dst,
            drought_df=drought_df,
            drought_multiplier=args.drought_multiplier,
        )
        rec["condition"] = condition
        rec["fold"] = fold
        records.append(rec)

        copy_if_exists(val_src, val_dst)
        copy_if_exists(test_src, test_dst)

    summary = pd.DataFrame(records)
    summary_path = out_root / "drought_augmented_split_summary.csv"
    summary.to_csv(summary_path, index=False)

    print(f"Condition: {condition}")
    print(f"Output root: {out_root}")
    print(f"Drought inventory rows: {len(drought_df)}")
    print(f"Drought multiplier: {args.drought_multiplier}")
    print(f"Wrote summary: {summary_path}")
    print()
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()