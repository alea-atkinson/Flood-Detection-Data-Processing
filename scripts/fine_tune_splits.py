from pathlib import Path
import random
import pandas as pd

# -------------------------------------------------
# User settings
# -------------------------------------------------

train_image_dir = Path("2025_Tile_Data/Only_PNG_Data")
train_mask_dir = Path("2025_Tile_Data/flood_change_mask_tiles")

test_image_dir = Path("milton/tiles")
test_mask_dir = Path("20m_files/Milton/masks")

validation_fraction = 0.20
seed = 42

output_dir = Path("training/fine_tune_csvs")
output_dir.mkdir(exist_ok=True)

# -------------------------------------------------
# Build training/validation list
# -------------------------------------------------

train_records = []

for i in range (1, 8):
    fp_dir = Path(f"{train_image_dir}/fp{i}")
    uav_dir = Path(f"{fp_dir}/UAVSAR")
    for image_path in sorted(uav_dir.glob("*.tif")):
        mask_path = Path(f"{fp_dir}/flood_mask/{image_path.name}")
        if not mask_path.exists():
            print(f"Missing mask for {image_path.name}")
            continue
        train_records.append({
            "uavsar_path": str(image_path),
            "flood_mask_path": str(mask_path)
        })

# Shuffle and split
random.seed(seed)
random.shuffle(train_records)

n_val = int(len(train_records) * validation_fraction)

val_records = train_records[:n_val]
train_records = train_records[n_val:]

# -------------------------------------------------
# Build testing list
# -------------------------------------------------

test_records = []

for i in range (1, 8):
    fp_dir = Path(f"{test_image_dir}/fp{i}")
    uav_dir = Path(f"{fp_dir}/uavsar")
    for image_path in sorted(uav_dir.glob("*.tif")):
        mask_path = Path(f"{fp_dir}/masks/{image_path.name}")
        if not mask_path.exists():
            print(f"Missing mask for {image_path.name}")
            continue
        test_records.append({
            "uavsar_path": str(image_path),
            "flood_mask_path": str(mask_path)
        })

# -------------------------------------------------
# Save CSVs
# -------------------------------------------------

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

print(f"Training tiles   : {len(train_records)}")
print(f"Validation tiles : {len(val_records)}")
print(f"Testing tiles    : {len(test_records)}")