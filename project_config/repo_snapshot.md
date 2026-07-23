# Repo Snapshot

Snapshot date: 2026-07-23

## Current Branch Context

- Active branch: `domain-balancing`
- Previous stable SSL branch: `ssl-lofpo`
- Repo root: `/mnt/linuxlab/home/reuuzheng/Flood-Detection-Data-Processing`
- Allowed working prefix: `/mnt/linuxlab/home/reuuzheng`

## Relevant Data Layout

The LOFPO split CSVs live under `training/lofo_csvs`.

Expected structure:

```text
training/lofo_csvs/
  heldout_fp1/
    train.csv
    validation.csv
    test.csv
  heldout_fp2/
    train.csv
    validation.csv
    test.csv
  ...
  heldout_fp7/
    train.csv
    validation.csv
    test.csv
```

Every split CSV is expected to contain `uavsar_path` and `flood_mask_path`.

## Historical Naming Quirk

`2025_Tile_Data/Only_PNG_Data` is not a guarantee about file extensions. It identifies the IEEE-valid tile subset originally associated with PNG-filtered train/test splits. Current CSV rows may point to `.tif` GeoTIFF files in this folder.

Any new script should branch on the actual file extension:

- `.tif` and `.tiff`: read with `rasterio`
- `.png`, `.jpg`, and `.jpeg`: read with `PIL`

## Mask and SAR Rules

- Florence flooded pixels: use `mask > 0` unless a dataset-specific nodata rule has been explicitly verified.
- Valid SAR pixels: at least one of the first 3 SAR channels is nonzero.

## Domain-Balancing Workspace

Domain-balancing experiment folders:

- `domain_balancing/scripts`
- `domain_balancing/tables`
- `domain_balancing/figures`
- `domain_balancing/summaries`
- `domain_balancing/logs`
- `domain_balancing/models`
- `domain_balancing/raw_results`

Git should ignore generated logs, model artifacts, and raw results:

- `domain_balancing/logs/`
- `domain_balancing/models/`
- `domain_balancing/raw_results/`
