import os
import rasterio
import numpy as np
import xarray as xr
from rasterio.windows import from_bounds
from rasterio.warp import calculate_default_transform, reproject, Resampling
from pyproj import Transformer
import logging
import re
import pandas as pd

# === Utility Functions ===
def add_buffer(bbox, buffer_deg):
    minlon, minlat, maxlon, maxlat = bbox
    return (minlon - buffer_deg, minlat - buffer_deg, maxlon + buffer_deg, maxlat + buffer_deg)

def project_bbox_to_target_bounds(bbox_wgs84, target_crs="EPSG:32637", resolution=1000):
    transformer = Transformer.from_crs("EPSG:4326", target_crs, always_xy=True)
    xmin, ymin = transformer.transform(bbox_wgs84[0], bbox_wgs84[1])
    xmax, ymax = transformer.transform(bbox_wgs84[2], bbox_wgs84[3])

    xmin = int(xmin // resolution) * resolution
    xmax = int(np.ceil(xmax / resolution)) * resolution
    ymin = int(ymin // resolution) * resolution
    ymax = int(np.ceil(ymax / resolution)) * resolution

    return (xmin, ymin, xmax, ymax)

def reproject_and_crop(src, target_crs, target_bounds, resolution):
    dst_transform, width, height = calculate_default_transform(
        src.crs, target_crs, src.width, src.height, *src.bounds, resolution=(resolution, resolution)
    )

    # Create output array
    dtype = src.dtypes[0]
    dst_data = np.full((height, width), np.nan, dtype=np.float32)

    reproject(
                source=rasterio.band(src, 1),
                destination=dst_data,
                src_transform=src.transform,
                src_crs=src.crs,
                dst_transform=dst_transform,
                dst_crs=target_crs,
                resampling=Resampling.bilinear,
                src_nodata=0,             # specify original nodata
                dst_nodata=np.nan         # convert to NaN
            )
    # Define crop window in target CRS
    window = from_bounds(*target_bounds, transform=dst_transform)
    window = window.round_offsets().round_shape()
    cropped = dst_data[
        int(window.row_off):int(window.row_off + window.height),
        int(window.col_off):int(window.col_off + window.width)
    ]

    cropped = np.where(cropped == dst_nodata, np.nan, cropped) # Replace nodata values with np.nan

    return cropped, dst_transform

# === Setup ===
bbox_jd = (39.0053612,21.4210088,39.3253612,21.7410088)
bbox = add_buffer(bbox_jd, 0.05)
target_crs = "EPSG:32637"
resolution = 1000
target_bounds = project_bbox_to_target_bounds(bbox, target_crs=target_crs, resolution=resolution)

script_dir = os.path.dirname(os.path.abspath(__file__))
input_folder = os.path.normpath(os.path.join(script_dir, "../sat_data/raw/Jeddah/landsat/"))
output_file = os.path.normpath(os.path.join(script_dir, "../sat_data/processed/Jeddah_landsat.nc"))
log_file = "crop_errors.log"

logging.basicConfig(filename=log_file, level=logging.ERROR, format='%(asctime)s - %(levelname)s - %(message)s')

# === Process ===
band_dataarrays_list = {}

for fname in sorted(os.listdir(input_folder)):
    if not fname.lower().endswith(".tif"):
        continue

    fpath = os.path.join(input_folder, fname)
    try:
        with rasterio.open(fpath) as src:
            match = re.search(r'^[^_]+_[^_]+_[^_]+_(\d{8})_.*_([a-zA-Z0-9_]+)\.tif$', fname)
            if not match:
                logging.warning(f"Unparsable filename: {fname}")
                continue

            date_str = match.group(1)
            band_name = match.group(2).lower()
            acquisition_time = pd.to_datetime(f"{date_str[:4]}-{date_str[4:6]}-{date_str[6:8]}")

            if src.crs is None:
                logging.error(f"No CRS for {fname}")
                continue

            cropped, dst_transform = reproject_and_crop(src, target_crs, target_bounds, resolution)

            if cropped.size == 0:
                logging.warning(f"Empty cropped result for {fname}")
                continue

            current_da = xr.DataArray(
                            cropped[np.newaxis, :, :],
                            dims=["time", "y", "x"],
                            coords={"time": [acquisition_time]},
                            name=band_name,
                            attrs={
                                "crs": target_crs,
                                "transform": dst_transform.to_gdal(),
                                "_FillValue": float(dst_nodata) if np.issubdtype(type(dst_nodata), np.integer) else dst_nodata
                            }
                        )

            band_dataarrays_list.setdefault(band_name, []).append(current_da)

    except Exception as e:
        logging.error(f"Error processing {fname}: {e}")

# === Combine ===
ds = xr.Dataset()
for band_name, da_list in band_dataarrays_list.items():
    if da_list:
        try:
            ds[band_name] = xr.concat(da_list, dim="time")
        except Exception as e:
            logging.error(f"Concat failed for {band_name}: {e}")

if ds:
    ds.to_netcdf(output_file)
    print(f"Combined dataset saved to: {output_file}")
else:
    print("No data processed.")
