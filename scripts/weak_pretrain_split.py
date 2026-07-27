from pathlib import Path
import random
import pandas as pd

# -------------------------------------------------
# User settings
# -------------------------------------------------
milton_dir = Path("milton/tiles")
neches_img_dir = Path("tiles/neches_16508_17089")
neches_mask_dir = Path("tiles/masks/neches_16508_17089")
levees_img_dir = Path("tiles/levees_35510_11033")
levees_mask_dir = Path("tiles/masks/levees")

validation_fraction = 0.20
seed = 42

output_dir = Path("training/fine_tune_csvs/pretrain_weak")
output_dir.mkdir(exist_ok=True)

# -------------------------------------------------
# Build training/validation list
# -------------------------------------------------

train_records = []

for i in range (1, 8):
    fp_dir = Path(f"{milton_dir}/fp{i}")
    uav_dir = Path(f"{fp_dir}/uavsar")
    for image_path in sorted(uav_dir.glob("*.tif")):
        mask_path = Path(f"{fp_dir}/masks/{image_path.name}")
        if not mask_path.exists():
            print(f"Missing mask for {image_path.name}")
            continue
        train_records.append({
            "uavsar_path": str(image_path),
            "flood_mask_path": str(mask_path)
        })

for image_path in sorted(neches_img_dir.glob("*.tif")):
        mask_path = Path(f"{neches_mask_dir}/{image_path.stem}_flood_mask.tif")
        if not mask_path.exists():
            print(f"Missing mask for {image_path.name}")
            continue
        train_records.append({
            "uavsar_path": str(image_path),
            "flood_mask_path": str(mask_path)
        })

for image_path in sorted(levees_img_dir.glob("*.tif")):
        mask_path = Path(f"{levees_mask_dir}/{image_path.stem}_flood_mask.tif")
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

print(f"Training tiles   : {len(train_records)}")
print(f"Validation tiles : {len(val_records)}")
