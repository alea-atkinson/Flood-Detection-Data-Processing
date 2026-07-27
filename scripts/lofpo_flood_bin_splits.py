from pathlib import Path
import csv
import numpy as np
import rasterio
import os
# -----------------------------
# Input / Output
# -----------------------------
fp = 7
input_csv = Path(f"training/fine_tune_csvs/no_overlap_converted/held_out_fp{fp}_test.csv")  # All Florence tiles
output_dir = Path(f"training/fine_tune_csvs/florence_lofpo_flood_bin_tests/fp{fp}")
os.makedirs(output_dir, exist_ok=True)
targets = [0, 0.01, 0.02, 0.05, 0.1, 0.25, 0.5, 100]




# -----------------------------
# Separate tiles into target bin
# and all other flood percentages
# -----------------------------
for lower, upper in zip(targets[:-1], targets[1:]):
    test_records = []
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
                test_records.append(row)


    fieldnames = ["uavsar_path", "flood_mask_path"]
    output_path = Path(output_dir / f"{lower}-{upper}.csv")
    # Training
    with open(output_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(test_records)


    # -----------------------------
    # Summary
    # -----------------------------

    print("========== Dataset Summary ==========")
    print(f"Output path: {output_path}")
    print(f"  Test: {len(test_records)}")
