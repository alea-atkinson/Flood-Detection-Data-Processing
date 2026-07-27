



import csv
from pathlib import Path


fp = 7

train_csv = Path(f"training/fine_tune_csvs/florence_lopfo/fp{fp}/train.csv")
val_csv = Path(f"training/fine_tune_csvs/florence_lopfo/fp{fp}/val.csv")
output_dir = Path(f"training/fine_tune_csvs/florence_lopfo/no_milton/fp{fp}")


train_records = []

with open(train_csv, newline="") as f:
    reader = csv.DictReader(f)

    for row in reader:

        if "milton" in row["uavsar_path"] :
            continue
        else:
            train_records.append(row)

val_records = []

with open(val_csv, newline="") as f:
    reader = csv.DictReader(f)

    for row in reader:

        if "milton" in row["uavsar_path"] :
            continue
        else:
            val_records.append(row)



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