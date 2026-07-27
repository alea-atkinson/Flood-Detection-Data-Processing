from pathlib import Path
import csv
import random

# -----------------------------
# Input / Output
# -----------------------------

root = Path("milton/tiles")
root_uavsar = Path("2025_Tile_Data/UAVSAR")
root_masks = Path("2025_Tile_Data/flood_change_mask_tiles")
output_dir = Path("training/fine_tune_csvs/no_overlap_converted/with_milton")





for i in range (1, 8):

    train_records = []
    val_records = []
    with open(f"training/fine_tune_csvs/strict_no_overlap/heldout_fp{i}_train_pool.csv", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            tile = row["tile_name"]
            image_path= root_uavsar / tile
            mask_path = root_masks / tile   
            train_records.append({
                "uavsar_path": str(image_path),
                "flood_mask_path": str(mask_path)
            })
    for j in range (1, 8):    
        fp_dir = Path(f"{root}/fp{j}")
        uav_dir = Path(f"{fp_dir}/uavsar")
        for image_path in sorted(uav_dir.glob("*.tif")):
            mask_path = Path(f"{fp_dir}/masks/{image_path.name}")
            if not mask_path.exists():
                print(f"Missing mask for {image_path.name}")
                continue
        
            train_records.append({
            "uavsar_path": str(image_path),
            "flood_mask_path": str(mask_path)})
      
    # Shuffle and split
    random.seed(42)
    random.shuffle(train_records)

    n_val = int(len(train_records) * 0.2)

    val_records = train_records[:n_val]
    train_records = train_records[n_val:]

    output_dir.mkdir(parents=True, exist_ok=True)

    train_output = output_dir / f"held_out_fp{i}_train.csv"
    val_output = output_dir / f"held_out_fp{i}_val.csv"
    

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




