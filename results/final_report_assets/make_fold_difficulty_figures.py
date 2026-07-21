import csv
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


CSV_PATH = Path("results/lofo_fold_difficulty_summary.csv")
OUT_DIR = Path("results/final_report_assets")
OUT_DIR.mkdir(parents=True, exist_ok=True)


def read_rows(path):
    with path.open(newline="", encoding="utf-8-sig") as f:
        rows = list(csv.DictReader(f))
    if not rows:
        raise ValueError(f"No rows found in {path}")
    return rows


rows = read_rows(CSV_PATH)

fps = [row["heldout_fp"] for row in rows]
weak_minus_random = [float(row["weak_minus_random_dice"]) for row in rows]
mean_flood_fraction = [float(row["mean_tile_flood_fraction"]) for row in rows]

# Figure 1: bar chart of weak - random Dice by fold
fig, ax = plt.subplots(figsize=(8, 5))
ax.bar(fps, weak_minus_random)
ax.axhline(0.0, linewidth=1)
ax.set_ylabel("Weak SimSiam Dice - Random Dice")
ax.set_xlabel("Held-out flight path")
ax.set_title("Weak SimSiam improvement by held-out flight path")
fig.tight_layout()
fig.savefig(OUT_DIR / "weak_minus_random_dice_by_fold.png", dpi=300, bbox_inches="tight")
plt.close(fig)

# Figure 2: scatter of flood fraction vs weak - random Dice
fig, ax = plt.subplots(figsize=(7, 5))
ax.scatter(mean_flood_fraction, weak_minus_random)

for x, y, label in zip(mean_flood_fraction, weak_minus_random, fps):
    ax.annotate(label, (x, y), textcoords="offset points", xytext=(5, 5))

ax.axhline(0.0, linewidth=1)
ax.set_xlabel("Mean tile flood fraction")
ax.set_ylabel("Weak SimSiam Dice - Random Dice")
ax.set_title("Flood fraction vs Weak SimSiam improvement")
fig.tight_layout()
fig.savefig(OUT_DIR / "flood_fraction_vs_weak_minus_random_dice.png", dpi=300, bbox_inches="tight")
plt.close(fig)

print("Saved:")
print(OUT_DIR / "weak_minus_random_dice_by_fold.png")
print(OUT_DIR / "flood_fraction_vs_weak_minus_random_dice.png")