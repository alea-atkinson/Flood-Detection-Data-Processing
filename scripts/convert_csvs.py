from pathlib import Path
import csv

# -----------------------------
# Input / Output
# -----------------------------

root_uavsar = Path("2025_Tile_Data/UAVSAR")  # All Florence tiles
root_masks = Path("2025_Tile_Data/flood_change_mask_tiles")
output_dir = Path("training/fine_tune_csvs/no_overlap_converted")



records = ["train", "test", "validation"]

# -----------------------------
# Separate tiles into target bin
# and all other flood percentages
# -----------------------------

for i in range (1, 8):
    train_records = []
    val_records = []
    test_records = []
    for j in range (0, 3):
        with open(f"training/fine_tune_csvs/strict_no_overlap/heldout_fp{i}_{records[j]}.csv", newline="") as f:
            reader = csv.DictReader(f)

            for row in reader:
                tile = row["tile_name"]
                image_path= root_uavsar / tile
                mask_path = root_masks / tile
                if j == 0:
                    train_records.append({
                        "uavsar_path": str(image_path),
                        "flood_mask_path": str(mask_path)
                    })
                elif j == 1:
                     test_records.append({
                        "uavsar_path": str(image_path),
                        "flood_mask_path": str(mask_path)
                    })
                else:
                    val_records.append({
                        "uavsar_path": str(image_path),
                        "flood_mask_path": str(mask_path)
                    })

    output_dir.mkdir(parents=True, exist_ok=True)

    train_output = output_dir / f"held_out_fp{i}_train.csv"
    val_output = output_dir / f"held_out_fp{i}_val.csv"
    test_output = output_dir / f"held_out_fp{i}_test.csv"

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


