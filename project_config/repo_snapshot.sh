#!/usr/bin/env bash
set -euo pipefail

echo "===== Git ====="
git branch --show-current
git status --short
git log --oneline --decorate --graph --all --max-count=15

echo
echo "===== Top-level repo folders ====="
find . -maxdepth 1 -type d \
  -not -path "./.git" \
  -not -path "." \
  | sort

echo
echo "===== Important experiment folders ====="
find training domain_balancing seed_replication project_config -maxdepth 2 -type d 2>/dev/null \
  | grep -v "__pycache__" \
  | sort

echo
echo "===== LOFPO split CSVs ====="
find training/lofo_csvs -maxdepth 2 -type f -name "*.csv" | sort

echo
echo "===== LOFPO split row counts ====="
python3 - <<'PY'
from pathlib import Path
import pandas as pd

root = Path("training/lofo_csvs")
for fold_dir in sorted(root.glob("heldout_fp*")):
    print(fold_dir.name)
    for name in ["train.csv", "validation.csv", "test.csv"]:
        p = fold_dir / name
        if p.exists():
            df = pd.read_csv(p)
            print(f"  {name}: {len(df)} rows, columns={list(df.columns)}")
        else:
            print(f"  MISSING: {name}")
PY

echo
echo "===== Sample LOFPO paths ====="
python3 - <<'PY'
from pathlib import Path
import pandas as pd

p = Path("training/lofo_csvs/heldout_fp1/train.csv")
df = pd.read_csv(p)
print(df.head(3).to_string(index=False))
PY

echo
echo "===== File extension summary in LOFPO CSVs ====="
python3 - <<'PY'
from pathlib import Path
from collections import Counter
import pandas as pd

counter = Counter()
root = Path("training/lofo_csvs")
for csv_path in root.glob("heldout_fp*/*.csv"):
    df = pd.read_csv(csv_path)
    for col in df.columns:
        if "path" in col.lower():
            for value in df[col].dropna():
                counter[Path(str(value)).suffix.lower()] += 1

for ext, count in sorted(counter.items()):
    print(f"{ext or '[no extension]'}: {count}")
PY

echo
echo "===== Ignored large-output folders ====="
for d in domain_balancing/logs domain_balancing/models domain_balancing/raw_results seed_replication/logs seed_replication/models seed_replication/raw_results; do
    if [ -d "$d" ]; then
        count=$(find "$d" -type f | wc -l)
        echo "$d: $count files"
    else
        echo "$d: missing"
    fi
done