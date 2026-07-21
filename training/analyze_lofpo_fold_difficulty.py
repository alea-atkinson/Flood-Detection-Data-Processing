#!/usr/bin/env python3
from __future__ import annotations

import csv
from pathlib import Path
from statistics import mean, median

import numpy as np
from PIL import Image

try:
    import rasterio
except ImportError:
    rasterio = None


REPO_ROOT = Path(".")
SPLIT_ROOT = REPO_ROOT / "training" / "lofo_csvs"
RESULTS_DIR = REPO_ROOT / "results"
OUT_PATH = RESULTS_DIR / "lofo_fold_difficulty_summary.csv"

THRESHOLD_SUMMARY = RESULTS_DIR / "lofo_threshold_selected_summary_all_6_methods.csv"


ESRI_RGB_CLASSES = {
    "water": (26, 91, 171),
    "trees": (53, 130, 33),
    "flooded_vegetation": (135, 209, 158),
    "crops": (255, 219, 92),
    "built_area": (237, 2, 42),
    "bare_ground": (237, 233, 228),
    "snow_ice": (242, 250, 255),
    "clouds": (200, 200, 200),
    "rangeland": (198, 173, 141),
}


def read_csv_rows(path: Path) -> list[dict]:
    if not path.exists():
        raise FileNotFoundError(path)

    with path.open(newline="", encoding="utf-8-sig") as f:
        rows = list(csv.DictReader(f))

    if not rows:
        raise ValueError(f"No rows found in {path}")

    return rows


def resolve_path(path_str: str) -> Path:
    p = Path(path_str)
    if p.is_absolute():
        return p
    return (REPO_ROOT / p).resolve()


def find_column(row: dict, candidates: list[str]) -> str | None:
    lower_to_original = {k.lower(): k for k in row.keys()}
    for c in candidates:
        if c.lower() in lower_to_original:
            return lower_to_original[c.lower()]
    return None


def infer_land_cover_path(row: dict, uavsar_path: Path, mask_path: Path) -> Path | None:
    explicit_col = find_column(
        row,
        [
            "land_cover_path",
            "landcover_path",
            "land_cover",
            "landcover",
            "lc_path",
        ],
    )
    if explicit_col and row.get(explicit_col):
        p = resolve_path(row[explicit_col])
        if p.exists():
            return p

    candidates = []

    for base in [mask_path, uavsar_path]:
        s = str(base)
        replacements = [
            ("flood_mask", "land_cover"),
            ("flood_masks", "land_cover"),
            ("Flood_Mask", "Land_Cover"),
            ("flood", "land_cover"),
            ("mask", "land_cover"),
            ("UAVSAR", "land_cover"),
            ("uavsar", "land_cover"),
        ]

        for old, new in replacements:
            if old in s:
                candidates.append(Path(s.replace(old, new)))

    stem_candidates = [
        mask_path.with_name(mask_path.name.replace("flood_mask", "land_cover")),
        mask_path.with_name(mask_path.name.replace("mask", "land_cover")),
        uavsar_path.with_name(uavsar_path.name.replace("uavsar", "land_cover")),
        uavsar_path.with_name(uavsar_path.name.replace("UAVSAR", "Land_Cover")),
    ]
    candidates.extend(stem_candidates)

    for c in candidates:
        if c.exists():
            return c

    return None


def load_image(path: Path) -> np.ndarray:
    """
    Load PNG/TIF imagery robustly.

    PIL sometimes cannot read GeoTIFFs written by geospatial tools.
    For those files, fall back to rasterio.
    Returns:
      - H x W for single-band images
      - H x W x C for multi-band images
    """
    try:
        return np.array(Image.open(path))
    except Exception:
        if rasterio is None:
            raise RuntimeError(
                f"PIL could not read {path}, and rasterio is not installed. "
                "Install/use rasterio or convert the TIFFs to PNG."
            )

        with rasterio.open(path) as src:
            arr = src.read()  # C x H x W

        if arr.shape[0] == 1:
            return arr[0]

        return np.transpose(arr, (1, 2, 0))

def make_valid_mask(img: np.ndarray) -> np.ndarray:
    if img.ndim == 2:
        return img != 0
    return np.any(img[..., :3] != 0, axis=-1)


def make_flood_mask(mask: np.ndarray) -> np.ndarray:
    if mask.ndim == 3:
        mask = mask[..., 0]
    return mask > 0


def summarize_sar(img: np.ndarray, valid: np.ndarray) -> dict:
    if img.ndim == 2:
        img = img[..., None]

    img = img[..., :3].astype(np.float32)
    out = {}

    for ch in range(img.shape[-1]):
        vals = img[..., ch][valid]
        if vals.size == 0:
            out[f"sar_ch{ch}_mean"] = ""
            out[f"sar_ch{ch}_std"] = ""
            out[f"sar_ch{ch}_p1"] = ""
            out[f"sar_ch{ch}_p99"] = ""
        else:
            out[f"sar_ch{ch}_mean"] = float(np.mean(vals))
            out[f"sar_ch{ch}_std"] = float(np.std(vals))
            out[f"sar_ch{ch}_p1"] = float(np.percentile(vals, 1))
            out[f"sar_ch{ch}_p99"] = float(np.percentile(vals, 99))

    return out


def classify_land_cover_rgb(lc: np.ndarray, valid: np.ndarray) -> dict:
    out = {f"{name}_fraction": "" for name in ESRI_RGB_CLASSES.keys()}

    if lc.ndim == 2:
        vals = lc[valid]
        total = vals.size
        if total == 0:
            return out

        # Numeric ESRI class values.
        numeric_map = {
            "water": 1,
            "trees": 2,
            "flooded_vegetation": 4,
            "crops": 5,
            "built_area": 7,
            "bare_ground": 8,
            "snow_ice": 9,
            "clouds": 10,
            "rangeland": 11,
        }

        for name, value in numeric_map.items():
            out[f"{name}_fraction"] = float(np.sum(vals == value) / total)

        return out

    rgb = lc[..., :3].astype(np.float32)
    pixels = rgb[valid]

    if pixels.size == 0:
        return out

    class_names = list(ESRI_RGB_CLASSES.keys())
    class_colors = np.array([ESRI_RGB_CLASSES[name] for name in class_names], dtype=np.float32)

    # Nearest palette color.
    dists = ((pixels[:, None, :] - class_colors[None, :, :]) ** 2).sum(axis=2)
    labels = np.argmin(dists, axis=1)

    total = labels.size
    for idx, name in enumerate(class_names):
        out[f"{name}_fraction"] = float(np.sum(labels == idx) / total)

    return out


def summarize_fold(fp: str, threshold_rows_by_fp: dict) -> dict:
    test_csv = SPLIT_ROOT / f"heldout_{fp}" / "test.csv"
    rows = read_csv_rows(test_csv)

    first = rows[0]
    uavsar_col = find_column(first, ["uavsar_path", "image_path", "img_path", "sar_path", "path"])
    mask_col = find_column(first, ["flood_mask_path", "mask_path", "label_path", "target_path"])

    if uavsar_col is None:
        raise ValueError(f"Could not find UAVSAR/image column in {test_csv}. Columns: {list(first.keys())}")
    if mask_col is None:
        raise ValueError(f"Could not find flood mask column in {test_csv}. Columns: {list(first.keys())}")

    tile_flood_fracs = []
    tile_valid_fracs = []

    total_pixels = 0
    total_valid_pixels = 0
    total_flood_pixels_valid = 0

    sar_ch_values = {0: [], 1: [], 2: []}

    land_cover_accum = {f"{name}_fraction": [] for name in ESRI_RGB_CLASSES.keys()}
    land_cover_found_count = 0

    for row in rows:
        uavsar_path = resolve_path(row[uavsar_col])
        mask_path = resolve_path(row[mask_col])

        if not uavsar_path.exists():
            raise FileNotFoundError(uavsar_path)
        if not mask_path.exists():
            raise FileNotFoundError(mask_path)

        img = load_image(uavsar_path)
        mask = load_image(mask_path)

        valid = make_valid_mask(img)
        flood = make_flood_mask(mask)

        if flood.shape != valid.shape:
            raise ValueError(
                f"Shape mismatch for {uavsar_path.name}: image valid {valid.shape}, flood {flood.shape}"
            )

        valid_count = int(valid.sum())
        pixel_count = int(valid.size)
        flood_valid_count = int((flood & valid).sum())

        total_pixels += pixel_count
        total_valid_pixels += valid_count
        total_flood_pixels_valid += flood_valid_count

        valid_frac = valid_count / pixel_count if pixel_count else 0.0
        flood_frac = flood_valid_count / valid_count if valid_count else 0.0

        tile_valid_fracs.append(valid_frac)
        tile_flood_fracs.append(flood_frac)

        if img.ndim == 2:
            img3 = img[..., None]
        else:
            img3 = img[..., :3]

        img3 = img3.astype(np.float32)
        for ch in range(min(3, img3.shape[-1])):
            vals = img3[..., ch][valid]
            if vals.size:
                sar_ch_values[ch].append(vals)

        lc_path = infer_land_cover_path(row, uavsar_path, mask_path)
        if lc_path is not None and lc_path.exists():
            lc = load_image(lc_path)
            if lc.shape[:2] == valid.shape:
                lc_summary = classify_land_cover_rgb(lc, valid)
                for k, v in lc_summary.items():
                    if v != "":
                        land_cover_accum[k].append(float(v))
                land_cover_found_count += 1

    out = {
        "heldout_fp": fp,
        "num_test_tiles": len(rows),
        "total_pixels": total_pixels,
        "valid_pixel_fraction_total": total_valid_pixels / total_pixels if total_pixels else 0.0,
        "mean_tile_valid_fraction": mean(tile_valid_fracs),
        "median_tile_valid_fraction": median(tile_valid_fracs),
        "total_flood_fraction_valid_pixels": total_flood_pixels_valid / total_valid_pixels if total_valid_pixels else 0.0,
        "mean_tile_flood_fraction": mean(tile_flood_fracs),
        "median_tile_flood_fraction": median(tile_flood_fracs),
        "land_cover_tiles_found": land_cover_found_count,
    }

    for ch in range(3):
        vals = np.concatenate(sar_ch_values[ch]) if sar_ch_values[ch] else np.array([])
        if vals.size:
            out[f"sar_ch{ch}_mean"] = float(np.mean(vals))
            out[f"sar_ch{ch}_std"] = float(np.std(vals))
            out[f"sar_ch{ch}_p1"] = float(np.percentile(vals, 1))
            out[f"sar_ch{ch}_p99"] = float(np.percentile(vals, 99))
        else:
            out[f"sar_ch{ch}_mean"] = ""
            out[f"sar_ch{ch}_std"] = ""
            out[f"sar_ch{ch}_p1"] = ""
            out[f"sar_ch{ch}_p99"] = ""

    for k, values in land_cover_accum.items():
        out[k] = mean(values) if values else ""

    result_row = threshold_rows_by_fp.get(fp, {})
    for key in [
        "random_threshold",
        "weak_simsiam_threshold",
        "random_dice",
        "weak_simsiam_dice",
        "random_iou",
        "weak_simsiam_iou",
        "random_precision",
        "weak_simsiam_precision",
        "random_recall",
        "weak_simsiam_recall",
        "random_accuracy",
        "weak_simsiam_accuracy",
    ]:
        out[key] = result_row.get(key, "")

    if out["random_dice"] != "" and out["weak_simsiam_dice"] != "":
        out["weak_minus_random_dice"] = float(out["weak_simsiam_dice"]) - float(out["random_dice"])
    else:
        out["weak_minus_random_dice"] = ""

    if out["random_precision"] != "" and out["weak_simsiam_precision"] != "":
        out["weak_minus_random_precision"] = float(out["weak_simsiam_precision"]) - float(out["random_precision"])
    else:
        out["weak_minus_random_precision"] = ""

    if out["random_recall"] != "" and out["weak_simsiam_recall"] != "":
        out["weak_minus_random_recall"] = float(out["weak_simsiam_recall"]) - float(out["random_recall"])
    else:
        out["weak_minus_random_recall"] = ""

    return out


def main():
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    threshold_rows = read_csv_rows(THRESHOLD_SUMMARY)
    threshold_rows_by_fp = {row["heldout_fp"]: row for row in threshold_rows}

    rows = []
    for i in range(1, 8):
        fp = f"fp{i}"
        print(f"Analyzing {fp}")
        rows.append(summarize_fold(fp, threshold_rows_by_fp))

    fieldnames = list(rows[0].keys())

    with OUT_PATH.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print(f"Wrote {OUT_PATH}")

    print()
    print("Quick summary:")
    for row in rows:
        print(
            row["heldout_fp"],
            "flood_frac=",
            f"{float(row['mean_tile_flood_fraction']):.4f}",
            "valid_frac=",
            f"{float(row['mean_tile_valid_fraction']):.4f}",
            "weak-random dice=",
            f"{float(row['weak_minus_random_dice']):+.4f}",
        )

def quick_debug():
    """Run this instead of main() to diagnose mask values."""
    import csv
    test_csv = SPLIT_ROOT / "heldout_fp1" / "test.csv"
    with test_csv.open(newline="", encoding="utf-8-sig") as f:
        rows = list(csv.DictReader(f))
    
    first = rows[0]
    mask_col = find_column(first, ["flood_mask_path", "mask_path", "label_path", "target_path"])
    mask_path = resolve_path(first[mask_col])
    
    mask = load_image(mask_path)
    print(f"Shape: {mask.shape}, dtype: {mask.dtype}")
    print(f"Min: {mask.min()}, Max: {mask.max()}")
    unique, counts = np.unique(mask, return_counts=True)
    print(f"Unique values: {list(zip(unique.tolist(), counts.tolist()))}")
    
    # Test different thresholds
    for thresh in [0, 1, 127, 128, 254, 255]:
        flood = mask > thresh if mask.ndim == 2 else mask[..., 0] > thresh
        print(f"> {thresh}: {flood.sum()} pixels ({100*flood.sum()/mask.size:.2f}%)")

# quick_debug()  # Uncomment this and comment out main() to run debug

if __name__ == "__main__":
    #quick_debug()
    main()