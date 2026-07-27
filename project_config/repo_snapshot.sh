#!/usr/bin/env bash
set -euo pipefail

echo "===== Git ====="
git branch --show-current || true
git status --short || true
git log --oneline --decorate --graph --all --max-count=15 || true

echo
echo "===== Top-level repo folders ====="
find . -maxdepth 1 -type d \
  -not -path "./.git" \
  -not -path "." \
  | sort

echo
echo "===== Important experiment folders ====="
for d in training domain_balancing seed_replication project_config logs models results; do
    if [ -d "$d" ]; then
        find "$d" -maxdepth 2 -type d 2>/dev/null
    else
        echo "$d: missing"
    fi
done | grep -v "__pycache__" | sort

echo
echo "===== Training split CSVs ====="
if [ -d training ]; then
    find training -type f \( \
      -name "train.csv" -o \
      -name "val.csv" -o \
      -name "validation.csv" -o \
      -name "test.csv" \
    \) | sort
else
    echo "training: missing"
fi

echo
echo "===== Training split row counts and columns ====="
python3 - <<'PY'
import csv
from pathlib import Path

split_names = {"train.csv", "val.csv", "validation.csv", "test.csv"}
root = Path("training")

if not root.exists():
    print("training: missing")
    raise SystemExit(0)

csv_paths = sorted(p for p in root.rglob("*.csv") if p.name in split_names)
if not csv_paths:
    print("No split CSVs found under training/")
    raise SystemExit(0)

for csv_path in csv_paths:
    try:
        with csv_path.open("r", encoding="utf-8-sig", newline="") as handle:
            reader = csv.reader(handle)
            columns = next(reader, [])
            rows = sum(1 for _ in reader)
    except Exception as exc:
        print(f"{csv_path}: ERROR reading CSV: {exc}")
        continue

    print(f"{csv_path}: {rows} rows, columns={columns}")
PY

echo
echo "===== Sample paths from training split CSVs ====="
python3 - <<'PY'
import csv
from pathlib import Path

split_names = {"train.csv", "val.csv", "validation.csv", "test.csv"}
preferred_path_columns = {
    "uavsar_path",
    "flood_mask_path",
    "mask_path",
    "image_path",
    "sar_path",
    "path",
}
root = Path("training")

if not root.exists():
    print("training: missing")
    raise SystemExit(0)

csv_paths = sorted(p for p in root.rglob("*.csv") if p.name in split_names)
printed = 0
for csv_path in csv_paths:
    try:
        with csv_path.open("r", encoding="utf-8-sig", newline="") as handle:
            reader = csv.DictReader(handle)
            columns = reader.fieldnames or []
            path_columns = [
                col for col in columns
                if col in preferred_path_columns or "path" in col.lower()
            ]
            if not path_columns:
                continue

            rows = []
            for row in reader:
                rows.append({col: row.get(col, "") for col in path_columns})
                if len(rows) >= 3:
                    break
    except Exception as exc:
        print(f"{csv_path}: ERROR reading CSV: {exc}")
        continue

    if not rows:
        continue

    printed += 1
    print(csv_path)
    for row in rows:
        print("  " + " | ".join(f"{col}={row[col]}" for col in path_columns))
    if printed >= 12:
        remaining = len(csv_paths) - printed
        if remaining > 0:
            print(f"... sample output capped after {printed} CSVs")
        break

if printed == 0:
    print("No path-like columns found in split CSVs under training/")
PY

echo
echo "===== File extension summary from training split CSV path columns ====="
python3 - <<'PY'
import csv
from pathlib import Path
from collections import Counter

counter = Counter()
by_column = Counter()
split_names = {"train.csv", "val.csv", "validation.csv", "test.csv"}
root = Path("training")

if root.exists():
    csv_paths = sorted(p for p in root.rglob("*.csv") if p.name in split_names)
else:
    csv_paths = []

for csv_path in csv_paths:
    try:
        with csv_path.open("r", encoding="utf-8-sig", newline="") as handle:
            reader = csv.DictReader(handle)
            columns = reader.fieldnames or []
            path_columns = [col for col in columns if "path" in col.lower()]
            for row in reader:
                for col in path_columns:
                    value = (row.get(col) or "").strip()
                    if not value:
                        continue
                    suffix = Path(value).suffix.lower() or "[no extension]"
                    counter[suffix] += 1
                    by_column[(col, suffix)] += 1
    except Exception as exc:
        print(f"{csv_path}: ERROR reading CSV: {exc}")

if not counter:
    print("No file extensions found in path-like columns.")
    raise SystemExit(0)

for ext, count in sorted(counter.items()):
    print(f"{ext}: {count}")

print()
print("By column:")
for (col, ext), count in sorted(by_column.items()):
    print(f"{col} {ext}: {count}")
PY

echo
echo "===== Ignored large-output folders ====="
for d in logs models results training/pretrain_weights domain_balancing/logs domain_balancing/models domain_balancing/raw_results seed_replication/logs seed_replication/models seed_replication/raw_results; do
    if [ -d "$d" ]; then
        count=$(find "$d" -type f | wc -l)
        echo "$d: $count files"
    else
        echo "$d: missing"
    fi
done
