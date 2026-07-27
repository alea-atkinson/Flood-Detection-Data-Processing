#!/usr/bin/env python3

"""
Create train/validation/test splits using only tiles with <=10% flooding.

Flood bins:
    0-1%
    1-2%
    2-5%
    5-10%

Each bin is split independently:
    40% Train
    10% Validation
    50% Test

Usage:
python split_sparse_floods.py \
    --input-csv master.csv \
    --output-dir sparse_flood_splits
"""

from pathlib import Path
import argparse
import csv
import random

import numpy as np
import rasterio


# ----------------------------
# Arguments
# ----------------------------

parser = argparse.ArgumentParser()

parser.add_argument("--input-csv", type=Path, default= "training/fine_tune_csvs/train_milton/test.csv")
parser.add_argument("--output-dir", type=Path, default= "training/fine_tune_csvs/flood_bin_splits_low_flood_training")
parser.add_argument("--seed", type=int, default=42)

args = parser.parse_args()

random.seed(args.seed)


# ----------------------------
# Read CSV
# ----------------------------

with args.input_csv.open("r", newline="") as f:
    reader = csv.DictReader(f)
    rows = list(reader)
    fieldnames = reader.fieldnames


# ----------------------------
# Flood percentage function
# ----------------------------

def flood_percent(mask_path):

    with rasterio.open(mask_path) as src:
        mask = src.read(1)

    valid = mask < 200

    if valid.sum() == 0:
        return None

    flood = np.logical_and(mask == 1, valid)

    return flood.sum() / valid.sum()


# ----------------------------
# Flood bins
# ----------------------------

bins = {
    "2-5": [],
    "5-10": [],
    "10-25": [],
}

print("Computing flood percentages...\n")

for row in rows:

    pct = flood_percent(row["flood_mask_path"])

    if pct is None:
        continue

    pct *= 100

    if pct < 1:
        continue

    elif pct < 2:
        continue

    elif pct < 5:
        bins["2-5"].append(row)

    elif pct < 10:
        bins["5-10"].append(row)

    elif pct < 25:
        bins["10-25"].append(row)

    # Ignore >10% flood


# ----------------------------
# Split each bin
# ----------------------------

train_rows = []
val_rows = []
test_rows = []

print("Split summary")
print("-" * 50)

for name, rows in bins.items():

    random.shuffle(rows)

    n = len(rows)

    n_train = int(0.60 * n)
    n_val = int(0.10 * n)

    train = rows[:n_train]
    val = rows[n_train:n_train + n_val]
    test = rows[n_train + n_val:]

    train_rows.extend(train)
    val_rows.extend(val)
    test_rows.extend(test)

    print(
        f"{name:>5}% : "
        f"{n:4d} total | "
        f"{len(train):4d} train | "
        f"{len(val):4d} val | "
        f"{len(test):4d} test"
    )


print("-" * 50)
print(f"Train: {len(train_rows)}")
print(f"Val:   {len(val_rows)}")
print(f"Test:  {len(test_rows)}")


# ----------------------------
# Save CSVs
# ----------------------------

args.output_dir.mkdir(parents=True, exist_ok=True)

for filename, subset in [
    ("train.csv", train_rows),
    ("val.csv", val_rows),
    ("test.csv", test_rows),
]:

    with (args.output_dir / filename).open("w", newline="") as f:

        writer = csv.DictWriter(f, fieldnames=fieldnames)

        writer.writeheader()
        writer.writerows(subset)

print("\nFinished.")
print(f"Saved to {args.output_dir}")