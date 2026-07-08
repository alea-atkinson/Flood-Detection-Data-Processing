import rasterio
import numpy as np
from rasterio.windows import Window
from rasterio.windows import transform as window_transform
import os

import argparse



def parse_args() -> argparse.Namespace:
    
    parser = argparse.ArgumentParser(description="Tile processed files.")
    parser.add_argument("--src-path", type=str, required=True)
    parser.add_argument("--dst-folder", type=str, required=True) 
    return parser.parse_args()

tile_size = 256

edge_count = 0
minimal_data_count = 0
total_tile_count = 0



args = parse_args()
with rasterio.open(args.src_path) as uav:

    tile_id = 0
    
    #tile by height, width
    for row in range(0, uav.height, tile_size):
        for col in range(0, uav.width, tile_size):
            #window size = 256
            window = Window(col, row, tile_size, tile_size)
            #read in data
            uavsar_tile = uav.read(window=window)

            # skip incomplete edge tiles (not the correct size)
            if uavsar_tile.shape != (3, tile_size, tile_size):
                edge_count +=1
                continue

            #compute how much of the tile has actual UAVSAR converage
            valid_pixels = np.any(uavsar_tile > 0, axis=0)
            valid_fraction = np.mean(valid_pixels)
            if valid_fraction < 0.5: #at least half contain data
                minimal_data_count += 1
                continue
            #preserve georeferencing data
            tile_transform = window_transform(
            window,
            uav.transform
        )
            #save valid tiles

            #uavsar
            tile_path = f"{args.dst_folder}/tile_{tile_id:05d}.tif" #format tile name with id as string

            with rasterio.open(
                tile_path,
                "w",
                driver="GTiff",
                height=tile_size,
                width=tile_size,
                count=3,
                dtype=uavsar_tile.dtype,
                crs=uav.crs,
                transform=tile_transform
            ) as dst:
                dst.write(uavsar_tile)
            
            total_tile_count +=1
            tile_id+=1

print(f"Edge Count: {edge_count}")
print(f"Minimal UAVSAR coverage Count: {minimal_data_count}")
print(f"Total tiles saved: {total_tile_count}")



