#!/usr/bin/env python3

from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt

TABLE_DIR = Path("drought_augmentation/tables")
FIG_DIR = Path("drought_augmentation/figures")
FIG_DIR.mkdir(parents=True, exist_ok=True)

df = pd.read_csv(TABLE_DIR / "init_comparison_by_fold.csv")

# Make fold ordering stable.
df["fold_num"] = df["fold"].str.replace("fp", "", regex=False).astype(int)
df = df.sort_values(["fold_num", "condition"])

pivot = df.pivot(index="fold", columns="condition", values="test_dice")
pivot = pivot.loc[[f"fp{i}" for i in range(1, 8)]]

ax = pivot[["random_init", "drought_negative_init"]].plot(kind="bar", figsize=(9, 5))
ax.set_title("Strict LOFPO Test Dice by Held-Out Flight Path")
ax.set_xlabel("Held-out flight path")
ax.set_ylabel("Test Dice")
ax.set_ylim(0, max(0.8, pivot.max().max() + 0.05))
ax.legend(["Random initialization", "Drought-negative initialization"])
plt.tight_layout()

out = FIG_DIR / "init_comparison_test_dice_by_fold.png"
plt.savefig(out, dpi=200)
print(f"Wrote {out}")

delta = pd.read_csv(TABLE_DIR / "init_comparison_paired_deltas.csv")
delta["fold_num"] = delta["fold"].str.replace("fp", "", regex=False).astype(int)
delta = delta.sort_values("fold_num")

ax = delta.plot(
    x="fold",
    y=[
        "test_dice_delta_drought_minus_random",
        "test_precision_delta_drought_minus_random",
        "test_recall_delta_drought_minus_random",
    ],
    kind="bar",
    figsize=(10, 5),
)
ax.axhline(0, linewidth=1)
ax.set_title("Drought-Negative Minus Random: Metric Deltas by Fold")
ax.set_xlabel("Held-out flight path")
ax.set_ylabel("Delta")
ax.legend(["Dice delta", "Precision delta", "Recall delta"])
plt.tight_layout()

out = FIG_DIR / "init_comparison_metric_deltas_by_fold.png"
plt.savefig(out, dpi=200)
print(f"Wrote {out}")
