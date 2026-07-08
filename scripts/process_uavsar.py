#!/usr/bin/env python3
"""
End-to-end UAVSAR preprocessing pipeline:

1. Enhanced Lee filtering
2. Reprojection/resampling to 20 m
3. 256x256 tiling
4. Validation of intermediate and final outputs

Example:
    python3 scripts/process_uavsar.py \
        --src raw_files/harvey/neches_16510_17089_015_170902_L090_CX_01_pauli.tif

By default, the target CRS is chosen automatically from the raster centroid
using the corresponding WGS84 UTM zone. Use --target-crs EPSG:xxxx to override.
"""

from __future__ import annotations

import argparse
import math
import shutil
from pathlib import Path

import numpy as np
import rasterio
from rasterio.enums import Resampling
from rasterio.warp import calculate_default_transform, reproject, transform_bounds
from rasterio.windows import Window
from rasterio.windows import transform as window_transform
from scipy.ndimage import uniform_filter


TARGET_RESOLUTION = 20.0
TILE_SIZE = 256
MIN_VALID_FRACTION = 0.5


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the complete UAVSAR preprocessing pipeline."
    )
    parser.add_argument(
        "--src",
        type=Path,
        required=True,
        help="Path to the raw 3-band Pauli GeoTIFF.",
    )
    parser.add_argument(
        "--target-crs",
        default="auto",
        help=(
            "Destination CRS, e.g. EPSG:32617. "
            "Default: auto-select WGS84 UTM zone from raster centroid."
        ),
    )
    parser.add_argument(
        "--name",
        default=None,
        help=(
            "Optional output stem. Default: first three underscore-separated "
            "tokens of the input filename, e.g. neches_16510_17089."
        ),
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Delete existing outputs for this scene and recompute all stages.",
    )
    return parser.parse_args()


def enhanced_lee(
    img: np.ndarray,
    win_size: int = 7,
    cu: float = 0.523,
    cmax: float = 1.73,
) -> np.ndarray:
    """Apply the same Enhanced Lee implementation as the existing lee_filter.py."""
    img = img.astype(np.float32)

    local_mean = uniform_filter(img, win_size)
    local_mean_sq = uniform_filter(img**2, win_size)
    local_var = np.maximum(local_mean_sq - local_mean**2, 0)
    local_std = np.sqrt(local_var)
    ci = local_std / (local_mean + 1e-8)

    result = np.zeros_like(img)

    mask1 = ci <= cu
    result[mask1] = local_mean[mask1]

    mask2 = ci >= cmax
    result[mask2] = img[mask2]

    mask3 = ~(mask1 | mask2)
    w = np.exp(
        -(ci[mask3] - cu) /
        (cmax - ci[mask3] + 1e-8)
    )
    result[mask3] = (
        local_mean[mask3] * w +
        img[mask3] * (1 - w)
    )

    return result


def derive_scene_name(src: Path, explicit_name: str | None) -> str:
    if explicit_name:
        return explicit_name

    tokens = src.stem.split("_")
    if len(tokens) < 3:
        raise ValueError(
            f"Cannot derive scene name from '{src.name}'. "
            "Pass --name explicitly."
        )
    return "_".join(tokens[:3])


def infer_event_name(src: Path) -> str | None:
    """
    Infer event folder from raw_files/<event>/<file>.tif.
    Returns None for raw_files/<file>.tif.
    """
    if src.parent.name == "raw_files":
        return None
    if src.parent.parent.name == "raw_files":
        return src.parent.name
    return None


def auto_utm_crs(src_path: Path) -> str:
    """Choose WGS84 UTM EPSG code from raster centroid in lon/lat."""
    with rasterio.open(src_path) as src:
        if src.crs is None:
            raise ValueError("Input raster has no CRS; cannot auto-select UTM zone.")

        bounds_wgs84 = transform_bounds(
            src.crs,
            "EPSG:4326",
            *src.bounds,
            densify_pts=21,
        )

    left, bottom, right, top = bounds_wgs84
    lon = (left + right) / 2.0
    lat = (bottom + top) / 2.0

    if not (-180 <= lon <= 180 and -80 <= lat <= 84):
        raise ValueError(
            f"Raster centroid ({lon:.4f}, {lat:.4f}) is outside normal UTM bounds."
        )

    zone = int(math.floor((lon + 180) / 6) + 1)
    epsg = 32600 + zone if lat >= 0 else 32700 + zone
    return f"EPSG:{epsg}"


def inspect_input(src_path: Path) -> None:
    if not src_path.exists():
        raise FileNotFoundError(f"Input file not found: {src_path}")
    if src_path.suffix.lower() not in {".tif", ".tiff"}:
        raise ValueError(f"Expected a TIFF file, got: {src_path.name}")

    with rasterio.open(src_path) as src:
        if src.count != 3:
            raise ValueError(
                f"Expected a 3-band Pauli TIFF, but found {src.count} bands."
            )
        if src.crs is None:
            raise ValueError("Input raster has no CRS.")
        if src.width < TILE_SIZE or src.height < TILE_SIZE:
            raise ValueError(
                f"Raster is only {src.width}x{src.height}; "
                f"smaller than one {TILE_SIZE}x{TILE_SIZE} tile."
            )

        print("Input validation passed:")
        print(f"  size: {src.width} x {src.height}")
        print(f"  bands: {src.count}")
        print(f"  dtype: {src.dtypes}")
        print(f"  CRS: {src.crs}")


def run_lee_filter(src_path: Path, dst_path: Path) -> None:
    print("\n[1/3] Enhanced Lee filtering")
    print(f"  input:  {src_path}")
    print(f"  output: {dst_path}")

    with rasterio.open(src_path) as src:
        data = src.read()
        profile = src.profile.copy()

    filtered = np.zeros_like(data, dtype=np.float32)

    for band in range(data.shape[0]):
        print(f"  filtering band {band + 1}/{data.shape[0]}")
        filtered[band] = enhanced_lee(data[band], win_size=7)

    profile.update(dtype="float32")

    with rasterio.open(dst_path, "w", **profile) as dst:
        dst.write(filtered)


def validate_filtered(src_path: Path, filtered_path: Path) -> None:
    if not filtered_path.exists():
        raise RuntimeError(f"Filtered output was not created: {filtered_path}")

    with rasterio.open(src_path) as src, rasterio.open(filtered_path) as out:
        assert out.count == src.count == 3, "Filtered band count changed."
        assert out.width == src.width and out.height == src.height, (
            "Filtered raster dimensions changed."
        )
        assert out.crs == src.crs, "Filtered raster CRS changed."
        assert all(dtype == "float32" for dtype in out.dtypes), (
            f"Expected float32 filtered output, got {out.dtypes}."
        )

    print("  validation: PASS")


def run_change_resolution(
    src_path: Path,
    dst_path: Path,
    target_crs: str,
    resolution: float = TARGET_RESOLUTION,
) -> None:
    print("\n[2/3] Reprojection and 20 m resampling")
    print(f"  input:      {src_path}")
    print(f"  output:     {dst_path}")
    print(f"  target CRS: {target_crs}")
    print(f"  resolution: {resolution} m")

    with rasterio.open(src_path) as src:
        transform, width, height = calculate_default_transform(
            src.crs,
            target_crs,
            src.width,
            src.height,
            *src.bounds,
            resolution=resolution,
        )

        profile = src.profile.copy()
        profile.update(
            crs=target_crs,
            transform=transform,
            width=width,
            height=height,
            dtype="float32",
        )

        with rasterio.open(dst_path, "w", **profile) as dst:
            for band in range(1, src.count + 1):
                print(f"  resampling band {band}/{src.count}")
                destination = np.empty((height, width), dtype=np.float32)

                reproject(
                    source=rasterio.band(src, band),
                    destination=destination,
                    src_transform=src.transform,
                    src_crs=src.crs,
                    dst_transform=transform,
                    dst_crs=target_crs,
                    resampling=Resampling.bilinear,
                )

                dst.write(destination, band)


def validate_resampled(
    resampled_path: Path,
    target_crs: str,
    resolution: float = TARGET_RESOLUTION,
) -> None:
    if not resampled_path.exists():
        raise RuntimeError(f"20 m output was not created: {resampled_path}")

    with rasterio.open(resampled_path) as src:
        assert src.count == 3, f"Expected 3 bands, got {src.count}."
        assert src.crs == rasterio.crs.CRS.from_string(target_crs), (
            f"CRS mismatch: expected {target_crs}, got {src.crs}."
        )

        xres, yres = src.res
        assert abs(xres - resolution) < 1e-6, (
            f"X resolution is {xres}, expected {resolution}."
        )
        assert abs(yres - resolution) < 1e-6, (
            f"Y resolution is {yres}, expected {resolution}."
        )

    print("  validation: PASS")


def run_tiling(
    src_path: Path,
    dst_folder: Path,
    tile_size: int = TILE_SIZE,
    min_valid_fraction: float = MIN_VALID_FRACTION,
) -> dict[str, int]:
    print("\n[3/3] Tiling")
    print(f"  input:  {src_path}")
    print(f"  output: {dst_folder}")
    print(f"  tile size: {tile_size} x {tile_size}")
    print(f"  minimum valid coverage: {min_valid_fraction:.0%}")

    dst_folder.mkdir(parents=True, exist_ok=True)

    edge_count = 0
    minimal_data_count = 0
    total_tile_count = 0

    with rasterio.open(src_path) as uav:
        tile_id = 0

        for row in range(0, uav.height, tile_size):
            for col in range(0, uav.width, tile_size):
                window = Window(col, row, tile_size, tile_size)
                uavsar_tile = uav.read(window=window)

                if uavsar_tile.shape != (3, tile_size, tile_size):
                    edge_count += 1
                    continue

                # Preserves the current pipeline's definition of coverage.
                valid_pixels = np.any(uavsar_tile > 0, axis=0)
                valid_fraction = float(np.mean(valid_pixels))

                if valid_fraction < min_valid_fraction:
                    minimal_data_count += 1
                    continue

                tile_transform = window_transform(window, uav.transform)
                tile_path = dst_folder / f"tile_{tile_id:05d}.tif"

                with rasterio.open(
                    tile_path,
                    "w",
                    driver="GTiff",
                    height=tile_size,
                    width=tile_size,
                    count=3,
                    dtype=uavsar_tile.dtype,
                    crs=uav.crs,
                    transform=tile_transform,
                ) as dst:
                    dst.write(uavsar_tile)

                total_tile_count += 1
                tile_id += 1

    stats = {
        "edge_count": edge_count,
        "minimal_data_count": minimal_data_count,
        "total_tile_count": total_tile_count,
    }

    print(f"  edge tiles skipped: {edge_count}")
    print(f"  low-coverage tiles skipped: {minimal_data_count}")
    print(f"  tiles saved: {total_tile_count}")

    return stats


def validate_tiles(
    src_path: Path,
    dst_folder: Path,
    expected_count: int,
    tile_size: int = TILE_SIZE,
) -> None:
    tile_paths = sorted(dst_folder.glob("tile_*.tif"))

    if len(tile_paths) != expected_count:
        raise RuntimeError(
            f"Expected {expected_count} tiles but found {len(tile_paths)}."
        )
    if not tile_paths:
        raise RuntimeError(
            "No tiles were produced. Check the input raster and coverage criterion."
        )

    with rasterio.open(src_path) as parent:
        parent_crs = parent.crs
        parent_dtype = parent.dtypes[0]

    for path in tile_paths:
        with rasterio.open(path) as tile:
            assert tile.width == tile_size and tile.height == tile_size, (
                f"{path.name} has size {tile.width}x{tile.height}."
            )
            assert tile.count == 3, f"{path.name} has {tile.count} bands."
            assert tile.crs == parent_crs, f"{path.name} CRS mismatch."
            assert tile.dtypes[0] == parent_dtype, f"{path.name} dtype mismatch."

    print("  validation: PASS")


def main() -> None:
    args = parse_args()
    src = args.src.resolve()

    inspect_input(src)

    repo_root = Path(__file__).resolve().parent.parent
    scene_name = derive_scene_name(src, args.name)
    event_name = infer_event_name(src)

    if event_name:
        filtered_dir = repo_root / "filtered_files" / event_name
        resampled_dir = repo_root / "20m_files" / event_name
        tiles_dir = repo_root / "tiles" / event_name / scene_name
    else:
        filtered_dir = repo_root / "filtered_files"
        resampled_dir = repo_root / "20m_files"
        tiles_dir = repo_root / "tiles" / scene_name

    filtered_dir.mkdir(parents=True, exist_ok=True)
    resampled_dir.mkdir(parents=True, exist_ok=True)
    tiles_dir.parent.mkdir(parents=True, exist_ok=True)

    filtered_path = filtered_dir / f"{scene_name}.tif"
    resampled_path = resampled_dir / f"{scene_name}.tif"

    target_crs = (
        auto_utm_crs(src)
        if args.target_crs.lower() == "auto"
        else args.target_crs
    )

    print("\nPipeline configuration:")
    print(f"  scene: {scene_name}")
    print(f"  event: {event_name or '(none)'}")
    print(f"  target CRS: {target_crs}")
    print(f"  filtered output: {filtered_path}")
    print(f"  20 m output: {resampled_path}")
    print(f"  tile folder: {tiles_dir}")

    outputs_exist = (
        filtered_path.exists()
        or resampled_path.exists()
        or tiles_dir.exists()
    )

    if outputs_exist and not args.overwrite:
        raise FileExistsError(
            "One or more outputs already exist. "
            "Use --overwrite to recompute this scene."
        )

    if args.overwrite:
        filtered_path.unlink(missing_ok=True)
        resampled_path.unlink(missing_ok=True)
        if tiles_dir.exists():
            shutil.rmtree(tiles_dir)

    run_lee_filter(src, filtered_path)
    validate_filtered(src, filtered_path)

    run_change_resolution(
        filtered_path,
        resampled_path,
        target_crs,
        TARGET_RESOLUTION,
    )
    validate_resampled(
        resampled_path,
        target_crs,
        TARGET_RESOLUTION,
    )

    stats = run_tiling(
        resampled_path,
        tiles_dir,
        TILE_SIZE,
        MIN_VALID_FRACTION,
    )
    validate_tiles(
        resampled_path,
        tiles_dir,
        stats["total_tile_count"],
        TILE_SIZE,
    )

    print("\nPipeline complete.")
    print(f"  filtered: {filtered_path}")
    print(f"  20 m:     {resampled_path}")
    print(f"  tiles:    {tiles_dir}")
    print(f"  count:    {stats['total_tile_count']}")


if __name__ == "__main__":
    main()
