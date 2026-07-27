import math


def utm_zone(lon):
    return int((lon + 180) // 6 + 1)


# UAVSAR bounds from ASF metadata
# Format: min_lon, min_lat, max_lon, max_lat
bounds = (
    -91.536880,  # min longitude
    31.914704,   # min latitude
    -90.780957,  # max longitude
    33.834515    # max latitude
)



min_lon, min_lat, max_lon, max_lat = bounds


print("UAVSAR bounds:")
print(f"Longitude: {min_lon} to {max_lon}")
print(f"Latitude: {min_lat} to {max_lat}")


# Determine UTM zones
z1 = utm_zone(min_lon)
z2 = utm_zone(max_lon)


if z1 == z2:
    print("Safe:", f"Entire flight path is in UTM zone {z1}")
else:
    print("Crosses zones:", z1, z2)