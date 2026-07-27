import rasterio

with rasterio.open("tiles/levees_35510_11033/tile_00000.tif") as src:
    print(src.crs)