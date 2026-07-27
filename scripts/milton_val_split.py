from pathlib import Path
import pandas as pd

# -------------------------------------------------
# User settings
# -------------------------------------------------

train_test_image_dir = Path("2025_Tile_Data/Only_PNG_Data")
val_image_dir = Path("milton/tiles")

# Choose which flight path to use for testing
test_fp = 7


output_dir = Path(f"training/fine_tune_csvs/milton_val/held_out_fp{test_fp}")


# -------------------------------------------------
# Build training/validation lists
# -------------------------------------------------

train_records = []
test_records = []
val_records = []

for i in range(1, 8):
    fp_dir = train_test_image_dir / f"fp{i}"
    uav_dir = fp_dir / "UAVSAR"
    mask_dir = fp_dir / "flood_mask"

    for image_path in sorted(uav_dir.glob("*.tif")):
        mask_path = mask_dir / image_path.name

        if not mask_path.exists():
            print(f"Missing mask for {image_path.name}")
            continue

        record = {
            "uavsar_path": str(image_path),
            "flood_mask_path": str(mask_path)
        }

        if i == test_fp:
            test_records.append(record)
        else:
            train_records.append(record)

# -------------------------------------------------
# Build testing list
# -------------------------------------------------



for i in range(1, 8):
    fp_dir = val_image_dir / f"fp{i}"
    uav_dir = fp_dir / "uavsar"
    mask_dir = fp_dir / "masks"

    for image_path in sorted(uav_dir.glob("*.tif")):
        mask_path = mask_dir / image_path.name

        if not mask_path.exists():
            print(f"Missing mask for {image_path.name}")
            continue

        val_records.append({
            "uavsar_path": str(image_path),
            "flood_mask_path": str(mask_path)
        })

# -------------------------------------------------
# Save CSVs
# -------------------------------------------------

output_dir.mkdir(parents=True, exist_ok=True)

pd.DataFrame(train_records).to_csv(
    output_dir / "train.csv",
    index=False
)

pd.DataFrame(val_records).to_csv(
    output_dir / "val.csv",
    index=False
)

pd.DataFrame(test_records).to_csv(
    output_dir / "test.csv",
    index=False
)

print(f"Test flight path : fp{test_fp}")
print(f"Training tiles         : {len(train_records)}")
print(f"Validation tiles       : {len(val_records)}")
print(f"Testing tiles          : {len(test_records)}")