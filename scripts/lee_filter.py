import numpy as np
import rasterio
from scipy.ndimage import uniform_filter
import argparse


def enhanced_lee(img, win_size=7, cu=0.523, cmax=1.73):
    img = img.astype(np.float32)

    local_mean = uniform_filter(img, win_size)

    local_mean_sq = uniform_filter(img**2, win_size)

    local_var = local_mean_sq - local_mean**2
    local_var = np.maximum(local_var, 0)

    local_std = np.sqrt(local_var)

    ci = local_std / (local_mean + 1e-8)

    result = np.zeros_like(img)

    # homogeneous
    mask1 = ci <= cu
    result[mask1] = local_mean[mask1]

    # heterogeneous
    mask2 = ci >= cmax
    result[mask2] = img[mask2]

    # intermediate
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


def parse_args() -> argparse.Namespace:
    
    parser = argparse.ArgumentParser(description="Apply lee filter.")
    parser.add_argument("--src-path", type=str, required=True)
    parser.add_argument("--dst-path", type=str, required=True) 
    return parser.parse_args()

args = parse_args()
input_uav = args.src_path
output_uav = args.dst_path


with rasterio.open(args.src_path) as src:
    data = src.read()
    profile = src.profile

filtered = np.zeros_like(data, dtype=np.float32) #needs to be converted to floating point for the lee filter

for band in range(data.shape[0]):
    print(f"Filtering band {band+1}")
    filtered[band] = enhanced_lee(
        data[band],
        win_size=7
    )

profile.update(dtype="float32") #matches florence datatype

with rasterio.open(
    args.dst_path,
    "w",
    **profile
) as dst:
    dst.write(filtered)