import rasterio
from rasterio.warp import reproject, Resampling, transform_bounds
from rasterio.windows import from_bounds
from pathlib import Path
import numpy as np
import os


UAVSAR_DIR = Path("tiles/levees_35510_11033")
NAIP_PATH = "naip/levees/naip_20m_uavsar.tif"

OUTPUT_DIR = Path("naip_tiles/levees")
OUTPUT_DIR.mkdir(exist_ok=True)


with rasterio.open(NAIP_PATH) as naip_src:

    print("NAIP CRS:", naip_src.crs)

    for uavsar_path in UAVSAR_DIR.glob("*.tif"):

        print(f"\nProcessing {uavsar_path.name}")

        with rasterio.open(uavsar_path) as sar_src:

            print("UAVSAR CRS:", sar_src.crs)

            # --------------------------------------------------
            # Reproject UAVSAR bounds into NAIP CRS
            # --------------------------------------------------
            sar_bounds = sar_src.bounds

            naip_bounds = transform_bounds(
                sar_src.crs,
                naip_src.crs,
                sar_bounds.left,
                sar_bounds.bottom,
                sar_bounds.right,
                sar_bounds.top
            )

            print("Original UAVSAR bounds:", sar_bounds)
            print("Transformed NAIP bounds:", naip_bounds)


            # --------------------------------------------------
            # Find corresponding area in NAIP raster
            # --------------------------------------------------
            naip_window = from_bounds(
                naip_bounds[0],
                naip_bounds[1],
                naip_bounds[2],
                naip_bounds[3],
                transform=naip_src.transform
            )

            naip_window = naip_window.round_offsets().round_lengths()

            print("NAIP window:", naip_window)


            # Check window is inside NAIP raster
            if (
                naip_window.width <= 0 or 
                naip_window.height <= 0
            ):
                print("WARNING: Empty NAIP window")
                continue


            # Read NAIP subset
            naip_data = naip_src.read(window=naip_window)


            if naip_data.size == 0:
                print("WARNING: Empty NAIP data")
                continue


            naip_transform = naip_src.window_transform(
                naip_window
            )


            # --------------------------------------------------
            # Create output matching UAVSAR exactly
            # --------------------------------------------------
            profile = {
                "driver": "GTiff",
                "height": sar_src.height,
                "width": sar_src.width,
                "count": naip_src.count,
                "dtype": naip_src.dtypes[0],
                "crs": sar_src.crs,
                "transform": sar_src.transform,
                "compress": "LZW",
                "nodata": 0
            }


            output_path = OUTPUT_DIR / uavsar_path.name


            with rasterio.open(output_path, "w", **profile) as dst:

                for band in range(1, naip_src.count + 1):

                    reproject(
                        source=naip_data[band-1],
                        destination=rasterio.band(dst, band),

                        src_transform=naip_transform,
                        src_crs=naip_src.crs,

                        dst_transform=sar_src.transform,
                        dst_crs=sar_src.crs,

                        resampling=Resampling.bilinear
                    )


            # --------------------------------------------------
            # Remove empty tiles
            # --------------------------------------------------
            with rasterio.open(output_path) as check:

                check_data = check.read()

                print(
                    "NAIP values:",
                    check_data.min(),
                    check_data.max()
                )

                if np.all(check_data == 0):
                    print(
                        f"Skipping empty tile: {uavsar_path.name}"
                    )
                    os.remove(output_path)
                    continue


            print("Saved:", output_path)


print("\nFinished NAIP tiling")