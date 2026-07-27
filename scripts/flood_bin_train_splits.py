from pathlib import Path
import csv
import numpy as np
import rasterio
import random

# -----------------------------
# Input / Output
# -----------------------------

input_csv = Path("training/fine_tune_csvs/train_milton/test.csv") #all florence tiles as input
output_dir = Path("training/fine_tune_csvs/train_flood_bin_splits/train_florence_10-50") 

# Keep only tiles with >theshold flood
lower = 0.10
upper = 0.50

train_records = []

# -----------------------------
# Filter tiles
# -----------------------------

with open(input_csv, newline="") as f:
    reader = csv.DictReader(f)

    for row in reader:

        # Read SAR
        with rasterio.open(row["uavsar_path"]) as src:
            sar = src.read()[:3]

        # Valid SAR pixels
        sar_valid = ~(sar == 0).all(axis=0)

        # Read mask
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

        if flood_fraction >= lower and flood_fraction < upper:
            train_records.append(row)

# Shuffle and split
random.seed(42)
random.shuffle(train_records)

n_val = int(len(train_records) * 0.2) #validation fraction = 0.2

val_records = train_records[:n_val]
train_records = train_records[n_val:]

train_output = Path(output_dir / 'train.csv')
val_output= Path(output_dir / 'val.csv')

train_output.parent.mkdir(parents=True, exist_ok=True)
val_output.parent.mkdir(parents=True, exist_ok=True)


#training
with open(train_output, "w", newline="") as f:
    writer = csv.DictWriter(
        f,
        fieldnames=["uavsar_path", "flood_mask_path"]
    )

    writer.writeheader()
    writer.writerows(train_records)


#validation
with open(val_output , "w", newline="") as f:
    writer = csv.DictWriter(
        f,
        fieldnames=["uavsar_path", "flood_mask_path"]
    )

    writer.writeheader()
    writer.writerows(val_records)

print(f"Saved {len(train_records)} tiles for training.")
print(f"Saved {len(val_records)} tiles for validation.")

print(f"Output Folder: {output_dir}")