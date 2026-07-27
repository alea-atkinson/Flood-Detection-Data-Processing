# No-Overlap Converted Split Audit Notes

Canonical split root:

`training/fine_tune_csvs/no_overlap_converted`

## Main finding

The `no_overlap_converted` split directory contains the canonical Florence-only and Florence+Milton split files for the next drought augmentation phase.

The base directory contains Florence-only train/validation/test files for held-out fp1 through fp7:

- `held_out_fp*_train.csv`
- `held_out_fp*_val.csv`
- `held_out_fp*_test.csv`

The `with_milton/` subdirectory contains Florence+Milton train/validation files:

- `with_milton/held_out_fp*_train.csv`
- `with_milton/held_out_fp*_val.csv`

There are no separate `with_milton` test files. For Florence-held-out evaluation, the correct test files are the base test files:

- `held_out_fp*_test.csv`

## File existence

The audit found no missing SAR files and no missing flood-mask files.

## Duplicate rows

The held-out test CSVs contain no duplicate SAR/mask pairs.

The train and validation CSVs contain repeated SAR/mask pairs. These appear to be exact duplicate rows, not merely similar tile names.

For baseline comparability with the current SSL branch, the drought augmentation experiments will preserve these existing duplicate rows rather than deduplicating the baseline splits.

## Drought experiment implication

For the first drought experiment:

- keep validation CSVs unchanged
- keep test CSVs unchanged
- add drought assumed-negative tiles only to train CSVs
- preserve existing baseline duplicate rows for comparability
- report duplicate counts as a known split property