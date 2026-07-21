import csv
from pathlib import Path

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt


OUT = Path("results/final_report_assets")
OUT.mkdir(parents=True, exist_ok=True)


def read_csv(path):
    if not path.exists():
        raise FileNotFoundError(f"Missing file: {path}")

    with open(path, newline="", encoding="utf-8-sig") as f:
        rows = list(csv.DictReader(f))

    if not rows:
        raise ValueError(f"No rows found in {path}")

    print(f"Loaded {len(rows)} rows from {path}")
    print("Columns:", rows[0].keys())

    return rows


method_rows = read_csv(OUT / "threshold_selected_method_summary.csv")
fold_rows = read_csv(OUT / "weak_simsiam_vs_random_by_fold.csv")


# Figure 1: mean Dice by method
methods = [r["Method"] for r in method_rows]
dice = [float(r["Mean Dice"]) for r in method_rows]

fig, ax = plt.subplots(figsize=(10, 5))
ax.bar(methods, dice)
ax.set_ylabel("Mean Dice")
ax.set_title("Threshold-selected LOFPO mean Dice by method")
ax.set_ylim(0.50, 0.62)
ax.tick_params(axis="x", rotation=30)
for label in ax.get_xticklabels():
    label.set_horizontalalignment("right")
fig.tight_layout()
fig.savefig(OUT / "mean_dice_by_method.png", dpi=300, bbox_inches="tight")
plt.close(fig)


# Figure 2: mean IoU by method
iou = [float(r["Mean IoU"]) for r in method_rows]

fig, ax = plt.subplots(figsize=(10, 5))
ax.bar(methods, iou)
ax.set_ylabel("Mean IoU")
ax.set_title("Threshold-selected LOFPO mean IoU by method")
ax.set_ylim(0.35, 0.45)
ax.tick_params(axis="x", rotation=30)
for label in ax.get_xticklabels():
    label.set_horizontalalignment("right")
fig.tight_layout()
fig.savefig(OUT / "mean_iou_by_method.png", dpi=300, bbox_inches="tight")
plt.close(fig)


# Figure 3: fold-by-fold Dice random vs weak SimSiam
fps = [r["Held-out Flight Path"] for r in fold_rows]
random_dice = [float(r["Random Dice"]) for r in fold_rows]
weak_dice = [float(r["Weak SimSiam Dice"]) for r in fold_rows]

fig, ax = plt.subplots(figsize=(8, 5))
ax.plot(fps, random_dice, marker="o", label="Random")
ax.plot(fps, weak_dice, marker="o", label="Weak SimSiam")
ax.set_ylabel("Test Dice")
ax.set_title("Fold-by-fold threshold-selected Dice")
ax.set_ylim(0.45, 0.75)
ax.legend()
fig.tight_layout()
fig.savefig(OUT / "fold_by_fold_random_vs_weak_simsiam.png", dpi=300, bbox_inches="tight")
plt.close(fig)


print("Saved figures:")
print(OUT / "mean_dice_by_method.png")
print(OUT / "mean_iou_by_method.png")
print(OUT / "fold_by_fold_random_vs_weak_simsiam.png")