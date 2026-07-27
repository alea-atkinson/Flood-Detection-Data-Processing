from pathlib import Path
import random
import pandas as pd
import os

# -------------------------------------------------
# User settings
# -------------------------------------------------


image_dir = Path("milton/tiles/florence")


validation_fraction = 0.20
seed = 42
fp = 7

output_dir = Path(f"milton/florence_new_data_all")
os.makedirs(output_dir, exist_ok=True)

# -------------------------------------------------
# Build training/validation list
# -------------------------------------------------

train_records = []
test_records = []

for i in range (1, 8):
    
    fp_dir = Path(f"{image_dir}/fp{i}")
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

        test_records.append({
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

pd.DataFrame(test_records).to_csv(
    output_dir / "test.csv",
    index=False
)

print(f"Training tiles   : {len(train_records)}")
print(f"Validation tiles : {len(val_records)}")
print(f"Testing tiles    : {len(test_records)}")