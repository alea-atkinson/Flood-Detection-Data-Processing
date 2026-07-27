from pathlib import Path
import csv
import numpy as np
import rasterio
import random

# -----------------------------
# Input / Output
# -----------------------------

input_csv = Path("training/fine_tune_csvs/train_milton/test.csv")  # All Florence tiles
output_dir = Path("training/fine_tune_csvs/florence_flood_bin_tests/50-100")

# Target flood bin
lower = 0.50
upper = 2

target_records = []
other_records = []

# -----------------------------
# Separate tiles into target bin
# and all other flood percentages
# -----------------------------

with open(input_csv, newline="") as f:
    reader = csv.DictReader(f)

    for row in reader:

        # Read SAR
        with rasterio.open(row["uavsar_path"]) as src:
            sar = src.read()[:3]

        # Valid SAR pixels
        sar_valid = ~(sar == 0).all(axis=0)

        # Read flood mask
        with rasterio.open(row["flood_mask_path"]) as src:
            mask = src.read(1)

            if src.nodata is None:
                mask_valid = np.ones(mask.shape, dtype=bool)
            else:
                mask_valid = mask != src.nodata

        valid = sar_valid & mask_valid

        valid_pixels = valid.sum()

        if valid_pixels == 0:
            continue

        flood_pixels = ((mask == 1) & valid).sum()
        flood_fraction = flood_pixels / valid_pixels

        if lower <= flood_fraction < upper:
            target_records.append(row)
        else:
            other_records.append(row)

# -----------------------------
# Shuffle
# -----------------------------

random.seed(42)

random.shuffle(target_records)
random.shuffle(other_records)

# -----------------------------
# Split target flood bin
# 50% train
# 10% validation
# 40% test
# -----------------------------

n_target = len(target_records)

n_train_target = int(0.50 * n_target)
n_val_target = int(0.10 * n_target)

train_target = target_records[:n_train_target]
val_target = target_records[
    n_train_target:n_train_target + n_val_target
]
test_records = target_records[
    n_train_target + n_val_target:
]

# -----------------------------
# Split remaining flood bins
# 70% train
# 30% validation
# -----------------------------

n_other = len(other_records)

n_train_other = int(0.80 * n_other)

train_other = other_records[:n_train_other]
val_other = other_records[n_train_other:]

# -----------------------------
# Combine datasets
# -----------------------------

train_records = train_target + train_other
val_records = val_target + val_other

random.shuffle(train_records)
random.shuffle(val_records)

# -----------------------------
# Output files
# -----------------------------

output_dir.mkdir(parents=True, exist_ok=True)

train_output = output_dir / "train.csv"
val_output = output_dir / "val.csv"
test_output = output_dir / "test.csv"

fieldnames = ["uavsar_path", "flood_mask_path"]

# Training
with open(train_output, "w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=fieldnames)
    writer.writeheader()
    writer.writerows(train_records)

# Validation
with open(val_output, "w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=fieldnames)
    writer.writeheader()
    writer.writerows(val_records)

# Testing
with open(test_output, "w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=fieldnames)
    writer.writeheader()
    writer.writerows(test_records)

# -----------------------------
# Summary
# -----------------------------

print("========== Dataset Summary ==========")
print(f"Output folder: {output_dir}")

print("\nOverall")
print(f"  Train: {len(train_records)}")
print(f"  Validation: {len(val_records)}")
print(f"  Test: {len(test_records)}")

print(f"\nTarget flood bin ({lower:.0%}–{upper:.0%})")
print(f"  Total: {n_target}")
print(f"  Train: {len(train_target)} ({len(train_target)/n_target:.1%})")
print(f"  Validation: {len(val_target)} ({len(val_target)/n_target:.1%})")
print(f"  Test: {len(test"uavsar_path": str(image_path),
                        "flood_mask_path": str(mask_path)_records)} ({len(test_records)/n_target:.1%})")

print("\nOther flood bins")
print(f"  Total: {n_other}")
print(f"  Train: {len(train_other)} ({len(train_other)/n_other:.1%})")
print(f"  Validation: {len(val_other)} ({len(val_other)/n_other:.1%})")