#!/usr/bin/env python3
"""Validate project path conventions and LOFPO CSV schemas."""

from __future__ import annotations

import argparse
import csv
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


CONFIG_PATH = Path(__file__).with_name("paths.yaml")


def _parse_scalar(value: str) -> Any:
    value = value.strip()
    if not value:
        return ""
    if (value.startswith('"') and value.endswith('"')) or (
        value.startswith("'") and value.endswith("'")
    ):
        return value[1:-1]
    return value


def _minimal_yaml_load(path: Path) -> dict[str, Any]:
    """Load the simple YAML subset used by paths.yaml without external deps."""
    lines = path.read_text(encoding="utf-8").splitlines()
    root: dict[str, Any] = {}
    stack: list[tuple[int, Any]] = [(-1, root)]

    for line_no, raw_line in enumerate(lines, 1):
        if not raw_line.strip() or raw_line.lstrip().startswith("#"):
            continue

        indent = len(raw_line) - len(raw_line.lstrip(" "))
        stripped = raw_line.strip()

        while stack and indent <= stack[-1][0]:
            stack.pop()
        parent = stack[-1][1]

        if stripped.startswith("- "):
            item = _parse_scalar(stripped[2:])
            if not isinstance(parent, list):
                raise ValueError(f"{path}:{line_no}: list item has no list parent")
            parent.append(item)
            continue

        if ":" not in stripped:
            raise ValueError(f"{path}:{line_no}: expected key/value pair")

        key, value = stripped.split(":", 1)
        key = key.strip()
        value = value.strip()

        if value:
            if not isinstance(parent, dict):
                raise ValueError(f"{path}:{line_no}: scalar key has no dict parent")
            parent[key] = _parse_scalar(value)
            continue

        if not isinstance(parent, dict):
            raise ValueError(f"{path}:{line_no}: nested key has no dict parent")

        container: dict[str, Any] | list[Any] = {}
        parent[key] = container
        stack.append((indent, container))

        # If the next significant line is a list item at a deeper indent, convert
        # this dict to a list when that line is encountered.
        for next_raw in lines[line_no:]:
            if not next_raw.strip() or next_raw.lstrip().startswith("#"):
                continue
            next_indent = len(next_raw) - len(next_raw.lstrip(" "))
            if next_indent > indent and next_raw.strip().startswith("- "):
                list_container: list[Any] = []
                parent[key] = list_container
                stack[-1] = (indent, list_container)
            break

    return root


def load_config(path: Path) -> dict[str, Any]:
    try:
        import yaml  # type: ignore
    except ImportError:
        return _minimal_yaml_load(path)

    with path.open("r", encoding="utf-8") as handle:
        loaded = yaml.safe_load(handle)
    if not isinstance(loaded, dict):
        raise ValueError(f"{path} did not load as a mapping")
    return loaded


def as_repo_path(repo_root: Path, raw_path: str) -> Path:
    path = Path(raw_path)
    return path if path.is_absolute() else repo_root / path


def is_relative_to(path: Path, parent: Path) -> bool:
    try:
        path.resolve().relative_to(parent.resolve())
    except ValueError:
        return False
    return True


def check_required_columns(csv_path: Path, required_columns: set[str]) -> list[str]:
    with csv_path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        fieldnames = set(reader.fieldnames or [])
    return sorted(required_columns - fieldnames)


def sample_extensions(
    csv_path: Path,
    repo_root: Path,
    path_columns: list[str],
    sample_size: int,
) -> tuple[Counter[str], list[Path]]:
    extensions: Counter[str] = Counter()
    missing_files: list[Path] = []

    with csv_path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        for row_index, row in enumerate(reader):
            if row_index >= sample_size:
                break
            for column in path_columns:
                raw_value = (row.get(column) or "").strip()
                if not raw_value:
                    extensions["<empty>"] += 1
                    continue
                data_path = as_repo_path(repo_root, raw_value)
                suffix = data_path.suffix.lower() or "<none>"
                extensions[suffix] += 1
                if not data_path.exists():
                    missing_files.append(data_path)

    return extensions, missing_files


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Validate repo path conventions and LOFPO split CSVs."
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=CONFIG_PATH,
        help="Path to paths.yaml.",
    )
    parser.add_argument(
        "--sample-size",
        type=int,
        default=25,
        help="Rows to sample from each split CSV when reporting file extensions.",
    )
    args = parser.parse_args()

    config = load_config(args.config)
    repo_root = Path(config["repo"]["root"])
    workspace_root = Path(config["repo"]["workspace_root"])
    lofpo_csv_root = as_repo_path(repo_root, config["data"]["lofpo_csv_root"])
    folds = config["data"]["lofpo_folds"]
    split_files = config["data"]["split_files"]
    required_columns = set(config["data"]["required_csv_columns"])
    path_columns = ["uavsar_path", "flood_mask_path"]

    errors: list[str] = []
    extension_totals: dict[str, Counter[str]] = defaultdict(Counter)

    if not repo_root.exists():
        errors.append(f"Repo root does not exist: {repo_root}")
    if not is_relative_to(repo_root, workspace_root):
        errors.append(f"Repo root is outside workspace root: {repo_root}")
    if not lofpo_csv_root.exists():
        errors.append(f"LOFPO CSV root does not exist: {lofpo_csv_root}")

    expected_split_names = {"train": "train.csv", "validation": "validation.csv", "test": "test.csv"}
    if split_files != expected_split_names:
        errors.append(
            "Unexpected split file names in paths.yaml; expected "
            f"{expected_split_names}, observed {split_files}"
        )

    for fold in folds:
        fold_dir = lofpo_csv_root / fold
        if not fold_dir.exists():
            errors.append(f"Missing fold directory: {fold_dir}")
            continue

        for split_name in ("train", "validation", "test"):
            csv_path = fold_dir / split_files[split_name]
            label = f"{fold}/{split_files[split_name]}"
            if not csv_path.exists():
                errors.append(f"Missing split CSV: {csv_path}")
                continue

            missing_columns = check_required_columns(csv_path, required_columns)
            if missing_columns:
                errors.append(f"{csv_path} missing columns: {', '.join(missing_columns)}")
                continue

            extensions, missing_files = sample_extensions(
                csv_path=csv_path,
                repo_root=repo_root,
                path_columns=path_columns,
                sample_size=args.sample_size,
            )
            extension_totals[label].update(extensions)
            if missing_files:
                preview = ", ".join(str(path) for path in missing_files[:5])
                extra = "" if len(missing_files) <= 5 else f" and {len(missing_files) - 5} more"
                errors.append(f"{label} sampled rows reference missing files: {preview}{extra}")

    print(f"Config: {args.config}")
    print(f"Repo root: {repo_root}")
    print(f"LOFPO CSV root: {lofpo_csv_root}")
    print(f"Sample size per CSV: {args.sample_size}")
    print()
    print("Observed sampled file extensions:")
    for label in sorted(extension_totals):
        observed = ", ".join(
            f"{extension}={count}" for extension, count in sorted(extension_totals[label].items())
        )
        print(f"  {label}: {observed or '<no sampled rows>'}")

    if errors:
        print()
        print("Validation failed:")
        for error in errors:
            print(f"  - {error}")
        return 1

    print()
    print("Validation passed.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
