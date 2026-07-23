# Data Dictionary

This file records path and naming rules that future scripts should treat as project conventions.

## Repository

- Repo root: `/mnt/linuxlab/home/reuuzheng/Flood-Detection-Data-Processing`
- Work must stay under `/mnt/linuxlab/home/reuuzheng`, not `/home`.
- Active branch: `domain-balancing`
- Previous stable SSL branch: `ssl-lofpo`

## LOFPO Splits

- LOFPO CSV root: `training/lofo_csvs`
- Fold directories: `heldout_fp1` through `heldout_fp7`
- Split CSV filenames:
  - `train.csv`
  - `validation.csv`
  - `test.csv`
- The validation split is named `validation.csv`, not `val.csv`.
- Required columns in every split CSV:
  - `uavsar_path`
  - `flood_mask_path`

Each CSV path is relative to the repo root unless explicitly documented otherwise.

## Tile Subset Name

The folder name `2025_Tile_Data/Only_PNG_Data` is historical and confusing.

It means the IEEE-valid tile subset originally associated with PNG-filtered train/test splits. It does not guarantee that all current files are PNGs. CSV paths may point to `.tif` GeoTIFF files.

Future scripts must inspect file extensions and use the appropriate reader instead of assuming a reader from this folder name.

## File Reading Rules

- Read `.tif` and `.tiff` files with `rasterio`.
- Read `.png`, `.jpg`, and `.jpeg` files with `PIL`.
- Do not assume `Only_PNG_Data` contains only PNG files.
- For Florence flood masks, treat flooded pixels as `mask > 0` unless a dataset-specific nodata rule is explicitly verified.
- Valid SAR pixels are pixels where at least one of the first 3 SAR channels is nonzero.

## Domain-Balancing Experiment Folders

- `domain_balancing/scripts`
- `domain_balancing/tables`
- `domain_balancing/figures`
- `domain_balancing/summaries`
- `domain_balancing/logs`
- `domain_balancing/models`
- `domain_balancing/raw_results`

The generated-output folders `domain_balancing/logs`, `domain_balancing/models`, and `domain_balancing/raw_results` should remain ignored by git.
