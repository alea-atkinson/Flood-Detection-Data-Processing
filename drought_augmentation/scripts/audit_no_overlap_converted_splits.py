#!/usr/bin/env python3
"""
Audit the canonical no-overlap converted split CSVs for the drought augmentation phase.

Canonical split root:
    training/fine_tune_csvs/no_overlap_converted

Expected layout:
    training/fine_tune_csvs/no_overlap_converted/
        held_out_fp1_train.csv
        held_out_fp1_val.csv
        held_out_fp1_test.csv
        ...
        held_out_fp7_train.csv
        held_out_fp7_val.csv
        held_out_fp7_test.csv

    training/fine_tune_csvs/no_overlap_converted/with_milton/
        held_out_fp1_train.csv
        held_out_fp1_val.csv
        ...
        held_out_fp7_train.csv
        held_out_fp7_val.csv

This script:
- summarizes CSV structure
- counts Florence/Milton/Drought/Unknown rows from paths
- infers source flight paths from path text
- checks whether SAR and mask files exist
- checks duplicate SAR/mask pairs
- optionally reads rasters and computes flood fraction with --deep

It does NOT modify any training data.
"""

from __future__ import annotations

import argparse
import re
from collections import Counter
from pathlib import Path
from typing import Any

import pandas as pd


UAVSAR_COL_CANDIDATES = [
    "uavsar_path",
    "sar_path",
    "image_path",
    "tile_path",
    "input_path",
]

MASK_COL_CANDIDATES = [
    "flood_mask_path",
    "mask_path",
    "label_path",
    "target_path",
]


def find_column(columns: list[str], candidates: list[str]) -> str | None:
    """Return the first matching column from candidates, case-insensitive."""
    lower_to_actual = {c.lower(): c for c in columns}
    for candidate in candidates:
        if candidate.lower() in lower_to_actual:
            return lower_to_actual[candidate.lower()]
    return None


def parse_split_csv_name(csv_path: Path, split_root: Path) -> tuple[str, str, str]:
    """
    Parse the no_overlap_converted CSV layout.

    Returns:
        condition, fold, split

    condition:
        florence_only or with_milton

    fold:
        fp1, fp2, ..., fp7

    split:
        train, val, test
    """
    rel = csv_path.relative_to(split_root)
    parts = rel.parts

    condition = "with_milton" if "with_milton" in parts else "florence_only"

    name = csv_path.name.lower()
    match = re.match(r"held_out_(fp\d+)_(train|val|validation|test)\.csv$", name)
    if match:
        fold = match.group(1)
        split = match.group(2)
        if split == "validation":
            split = "val"
        return condition, fold, split

    # Fallback for unexpected names.
    fold_match = re.search(r"(fp\d+)", name)
    fold = fold_match.group(1) if fold_match else csv_path.parent.name

    if "train" in name:
        split = "train"
    elif "val" in name:
        split = "val"
    elif "test" in name:
        split = "test"
    else:
        split = csv_path.stem

    return condition, fold, split


def infer_event_from_path(path: str) -> str:
    """Infer broad dataset/event name from a file path."""
    p = path.lower().replace("\\", "/")

    if "milton" in p or "florida" in p:
        return "milton"

    # On this branch, Florence paths commonly appear as:
    # 2025_Tile_Data/UAVSAR/...
    # 2025_Tile_Data/flood_change_mask_tiles/...
    if (
        "florence" in p
        or "only_png_data" in p
        or "2025_tile_data" in p
        or re.search(r"/fp[1-7]/", p)
    ):
        return "florence"

    if "drought" in p:
        return "drought"

    return "unknown"

def infer_source_fp_from_path(path: str) -> str:
    """
    Infer source flight path from path text.

    This detects path components like /fp1/, /fp2/, etc.
    """
    p = path.replace("\\", "/").lower()
    match = re.search(r"(?:^|/)fp([1-9][0-9]*)(?:/|$)", p)
    if match:
        return f"fp{match.group(1)}"
    return "unknown"


def resolve_path(repo_root: Path, raw_path: Any) -> Path | None:
    """Resolve a possibly relative path against repo root."""
    if pd.isna(raw_path):
        return None

    s = str(raw_path).strip()
    if not s:
        return None

    p = Path(s)
    if p.is_absolute():
        return p

    return repo_root / p


def extension_of(path: Any) -> str:
    if pd.isna(path):
        return ""
    return Path(str(path)).suffix.lower()


def safe_rel(path: Path, root: Path) -> str:
    try:
        return str(path.relative_to(root))
    except ValueError:
        return str(path)


def safe_read_raster_summary(path: Path) -> dict[str, Any]:
    """
    Read lightweight raster metadata and basic value summary.

    Requires rasterio. Used only with --deep.
    """
    try:
        import numpy as np
        import rasterio
    except ImportError:
        return {
            "read_ok": False,
            "read_error": "rasterio or numpy not installed",
        }

    try:
        with rasterio.open(path) as src:
            arr = src.read()
            summary: dict[str, Any] = {
                "read_ok": True,
                "read_error": "",
                "width": src.width,
                "height": src.height,
                "count": src.count,
                "dtype": ",".join(src.dtypes),
                "crs": str(src.crs) if src.crs else "",
                "nodata": src.nodata,
            }

            if arr.size <= 10_000_000:
                vals = np.unique(arr)
                if vals.size <= 20:
                    summary["unique_values"] = ",".join(str(v) for v in vals.tolist())
                else:
                    summary["unique_values"] = f"{vals.size} unique values"
            else:
                summary["unique_values"] = "skipped_large_array"

            return summary

    except Exception as e:
        return {
            "read_ok": False,
            "read_error": repr(e),
        }


def compute_flood_fraction(mask_path: Path, sar_path: Path | None = None) -> dict[str, Any]:
    """
    Compute flood fraction from mask.

    Rules:
    - mask == 255 is treated as nodata.
    - flooded pixels are mask > 0, excluding nodata.
    - if SAR is readable, valid SAR pixels are pixels where at least one of the first 3 channels is nonzero.
    """
    try:
        import numpy as np
        import rasterio
    except ImportError:
        return {
            "flood_fraction_ok": False,
            "flood_fraction_error": "rasterio or numpy not installed",
        }

    try:
        with rasterio.open(mask_path) as msrc:
            mask = msrc.read(1)

        mask_valid = mask != 255
        flood = (mask > 0) & mask_valid

        if sar_path is not None and sar_path.exists():
            try:
                with rasterio.open(sar_path) as ssrc:
                    band_count = min(3, ssrc.count)
                    sar = ssrc.read(indexes=list(range(1, band_count + 1)))
                sar_valid = np.any(sar != 0, axis=0)
                valid = mask_valid & sar_valid
            except Exception:
                valid = mask_valid
        else:
            valid = mask_valid

        total_valid = int(valid.sum())
        flood_pixels = int((flood & valid).sum())

        flood_fraction = None if total_valid == 0 else flood_pixels / total_valid

        return {
            "flood_fraction_ok": True,
            "flood_fraction_error": "",
            "valid_pixels": total_valid,
            "flood_pixels": flood_pixels,
            "flood_fraction": flood_fraction,
        }

    except Exception as e:
        return {
            "flood_fraction_ok": False,
            "flood_fraction_error": repr(e),
        }


def summarize_csv(
    csv_path: Path,
    repo_root: Path,
    split_root: Path,
    deep: bool,
    max_examples: int,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Summarize one split CSV and optionally produce per-row audit records."""
    df = pd.read_csv(csv_path)
    columns = list(df.columns)

    uavsar_col = find_column(columns, UAVSAR_COL_CANDIDATES)
    mask_col = find_column(columns, MASK_COL_CANDIDATES)

    condition, fold, split_name = parse_split_csv_name(csv_path, split_root)

    rel_csv = safe_rel(csv_path, repo_root)
    rel_to_split_root = safe_rel(csv_path, split_root)

    row_records: list[dict[str, Any]] = []

    event_counts: Counter[str] = Counter()
    fp_counts: Counter[str] = Counter()
    sar_ext_counts: Counter[str] = Counter()
    mask_ext_counts: Counter[str] = Counter()

    missing_sar = 0
    missing_mask = 0

    duplicate_pairs = 0
    if uavsar_col and mask_col:
        duplicate_pairs = int(df.duplicated(subset=[uavsar_col, mask_col]).sum())

    flood_fractions: list[float] = []

    for idx, row in df.iterrows():
        raw_sar = row[uavsar_col] if uavsar_col else None
        raw_mask = row[mask_col] if mask_col else None

        sar_str = "" if pd.isna(raw_sar) else str(raw_sar)
        mask_str = "" if pd.isna(raw_mask) else str(raw_mask)

        combined_path_text = sar_str + " " + mask_str
        event = infer_event_from_path(combined_path_text)
        source_fp = infer_source_fp_from_path(combined_path_text)

        event_counts[event] += 1
        fp_counts[source_fp] += 1
        sar_ext_counts[extension_of(sar_str)] += 1
        mask_ext_counts[extension_of(mask_str)] += 1

        sar_path = resolve_path(repo_root, raw_sar)
        mask_path = resolve_path(repo_root, raw_mask)

        sar_exists = bool(sar_path and sar_path.exists())
        mask_exists = bool(mask_path and mask_path.exists())

        if not sar_exists:
            missing_sar += 1
        if not mask_exists:
            missing_mask += 1

        record: dict[str, Any] = {
            "csv_path": rel_csv,
            "relative_to_split_root": rel_to_split_root,
            "condition": condition,
            "fold": fold,
            "split": split_name,
            "row_index": idx,
            "event": event,
            "source_fp": source_fp,
            "uavsar_path": sar_str,
            "flood_mask_path": mask_str,
            "sar_exists": sar_exists,
            "mask_exists": mask_exists,
            "sar_ext": extension_of(sar_str),
            "mask_ext": extension_of(mask_str),
        }

        if deep:
            if sar_path and sar_exists:
                sar_summary = safe_read_raster_summary(sar_path)
                for k, v in sar_summary.items():
                    record[f"sar_{k}"] = v

            if mask_path and mask_exists:
                mask_summary = safe_read_raster_summary(mask_path)
                for k, v in mask_summary.items():
                    record[f"mask_{k}"] = v

                frac_summary = compute_flood_fraction(mask_path, sar_path if sar_exists else None)
                record.update(frac_summary)

                if frac_summary.get("flood_fraction") is not None:
                    flood_fractions.append(float(frac_summary["flood_fraction"]))

        row_records.append(record)

    example_paths = []
    if uavsar_col and len(df) > 0:
        example_paths = df[uavsar_col].dropna().astype(str).head(max_examples).tolist()

    summary: dict[str, Any] = {
        "csv_path": rel_csv,
        "relative_to_split_root": rel_to_split_root,
        "condition": condition,
        "fold": fold,
        "split": split_name,
        "num_rows": len(df),
        "columns": "|".join(columns),
        "uavsar_col": uavsar_col or "",
        "mask_col": mask_col or "",
        "num_florence": event_counts.get("florence", 0),
        "num_milton": event_counts.get("milton", 0),
        "num_drought": event_counts.get("drought", 0),
        "num_unknown_event": event_counts.get("unknown", 0),
        "source_fp_counts": dict(fp_counts),
        "sar_ext_counts": dict(sar_ext_counts),
        "mask_ext_counts": dict(mask_ext_counts),
        "missing_sar_files": missing_sar,
        "missing_mask_files": missing_mask,
        "duplicate_pairs": duplicate_pairs,
        "unique_pairs": len(df) - duplicate_pairs,
        "duplicate_rate": duplicate_pairs / len(df) if len(df) else 0.0,
        "example_paths": " | ".join(example_paths),
    }

    if deep:
        if flood_fractions:
            summary["mean_flood_fraction"] = sum(flood_fractions) / len(flood_fractions)
            summary["min_flood_fraction"] = min(flood_fractions)
            summary["max_flood_fraction"] = max(flood_fractions)
        else:
            summary["mean_flood_fraction"] = ""
            summary["min_flood_fraction"] = ""
            summary["max_flood_fraction"] = ""

    return summary, row_records


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--repo-root",
        type=Path,
        default=Path("."),
        help="Repository root. Default: current directory.",
    )
    parser.add_argument(
        "--split-root",
        type=Path,
        default=Path("training/fine_tune_csvs/no_overlap_converted"),
        help="Canonical no-overlap split root to audit.",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path("drought_augmentation/tables"),
        help="Output directory for audit CSVs.",
    )
    parser.add_argument(
        "--deep",
        action="store_true",
        help="Read raster files and compute flood fractions. Slower.",
    )
    parser.add_argument(
        "--max-examples",
        type=int,
        default=3,
        help="Number of example paths to record per CSV.",
    )
    args = parser.parse_args()

    repo_root = args.repo_root.resolve()

    split_root = args.split_root
    if not split_root.is_absolute():
        split_root = repo_root / split_root
    split_root = split_root.resolve()

    out_dir = args.out_dir
    if not out_dir.is_absolute():
        out_dir = repo_root / out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    if not split_root.exists():
        raise FileNotFoundError(f"Split root does not exist: {split_root}")

    csv_paths = sorted(
        p
        for p in split_root.rglob("*.csv")
        if re.match(r"held_out_fp\d+_(train|val|validation|test)\.csv$", p.name.lower())
    )

    if not csv_paths:
        all_csvs = sorted(split_root.rglob("*.csv"))
        print(f"No held_out_fp*_train/val/test CSVs found under: {split_root}")
        print()
        print("All CSV files found:")
        for p in all_csvs[:200]:
            print(f"  {safe_rel(p, repo_root)}")
        raise FileNotFoundError(f"No usable split CSVs found under: {split_root}")

    summaries: list[dict[str, Any]] = []
    all_rows: list[dict[str, Any]] = []

    for csv_path in csv_paths:
        print(f"Auditing {safe_rel(csv_path, repo_root)}")
        summary, rows = summarize_csv(
            csv_path=csv_path,
            repo_root=repo_root,
            split_root=split_root,
            deep=args.deep,
            max_examples=args.max_examples,
        )
        summaries.append(summary)
        all_rows.extend(rows)

    summary_df = pd.DataFrame(summaries)
    rows_df = pd.DataFrame(all_rows)

    sort_cols = ["condition", "fold", "split"]
    summary_df = summary_df.sort_values(sort_cols).reset_index(drop=True)
    rows_df = rows_df.sort_values(sort_cols + ["row_index"]).reset_index(drop=True)

    summary_csv = out_dir / "no_overlap_converted_split_audit_summary.csv"
    rows_csv = out_dir / "no_overlap_converted_split_audit_rows.csv"
    text_summary = out_dir / "no_overlap_converted_split_audit_summary.txt"

    summary_df.to_csv(summary_csv, index=False)
    rows_df.to_csv(rows_csv, index=False)

    with text_summary.open("w", encoding="utf-8") as f:
        f.write("No-Overlap Converted Split Audit\n")
        f.write("================================\n\n")
        f.write(f"Repo root: {repo_root}\n")
        f.write(f"Split root: {split_root}\n")
        f.write(f"Deep raster inspection: {args.deep}\n")
        f.write(f"Number of split CSVs found: {len(csv_paths)}\n")
        f.write(f"Total rows across split CSVs: {int(summary_df['num_rows'].sum())}\n\n")

        f.write("Rows by CSV:\n")
        for _, row in summary_df.iterrows():
            f.write(
                f"- {row['relative_to_split_root']}: "
                f"condition={row['condition']}; "
                f"fold={row['fold']}; "
                f"split={row['split']}; "
                f"{row['num_rows']} rows; "
                f"Florence={row['num_florence']}, "
                f"Milton={row['num_milton']}, "
                f"Drought={row['num_drought']}, "
                f"Unknown={row['num_unknown_event']}; "
                f"missing SAR={row['missing_sar_files']}, "
                f"missing mask={row['missing_mask_files']}, "
                f"duplicates={row['duplicate_pairs']}, "
                f"unique_pairs={row['unique_pairs']}, "
                f"duplicate_rate={row['duplicate_rate']:.3f}"
            )

            if args.deep:
                f.write(
                    f"; mean flood fraction={row.get('mean_flood_fraction', '')}, "
                    f"min={row.get('min_flood_fraction', '')}, "
                    f"max={row.get('max_flood_fraction', '')}"
                )

            f.write("\n")

        f.write("\nCondition totals:\n")
        condition_totals = (
            summary_df.groupby("condition", dropna=False)[
                ["num_rows", "num_florence", "num_milton", "num_drought", "num_unknown_event"]
            ]
            .sum()
            .reset_index()
        )
        for _, row in condition_totals.iterrows():
            f.write(
                f"- {row['condition']}: "
                f"rows={row['num_rows']}, "
                f"Florence={row['num_florence']}, "
                f"Milton={row['num_milton']}, "
                f"Drought={row['num_drought']}, "
                f"Unknown={row['num_unknown_event']}\n"
            )

        f.write("\nOutputs:\n")
        f.write(f"- {safe_rel(summary_csv, repo_root)}\n")
        f.write(f"- {safe_rel(rows_csv, repo_root)}\n")
        f.write(f"- {safe_rel(text_summary, repo_root)}\n")

    print()
    print(f"Wrote summary CSV: {safe_rel(summary_csv, repo_root)}")
    print(f"Wrote row audit CSV: {safe_rel(rows_csv, repo_root)}")
    print(f"Wrote text summary: {safe_rel(text_summary, repo_root)}")


if __name__ == "__main__":
    main()