import rasterio
from rasterio.warp import calculate_default_transform, reproject
from rasterio.enums import Resampling
import numpy as np
import argparse


target_crs = "EPSG:32617" #need to use this crs because it is meter based insteead of degree based3
target_resolution = 20 #resolution used in IEEE AKA the NC data



def parse_args() -> argparse.Namespace:
    
    parser = argparse.ArgumentParser(description="Change UAVSAR resolution.")
    parser.add_argument("--src-path", type=str, required=True)
    parser.add_argument("--dst-path", type=str, required=True) 
    return parser.parse_args()

args = parse_args()
input_uav = args.src_path
output_uav = args.dst_path

with rasterio.open(input_uav) as src:

    transform, width, height = calculate_default_transform(
        src.crs,
        target_crs,
        src.width,
        src.height,
        *src.bounds,
        resolution=target_resolution
    )

    profile = src.profile.copy()

    profile.update(
        crs=target_crs,
        transform=transform,
        width=width,
        height=height
    )

    with rasterio.open(output_uav, "w", **profile) as dst:

        for band in range(1, src.count + 1):

            print(f"Processing band {band}")

            destination = np.empty(
                (height, width),
                dtype=np.float32
            )

            reproject(
                source=rasterio.band(src, band),
                destination=destination,
                src_transform=src.transform,
                src_crs=src.crs,
                dst_transform=transform,
                dst_crs=target_crs,
                resampling=Resampling.bilinear
            )

            dst.write(destination, band)

