import geetools
import geemap
import random
import rasterio
from rasterio.plot import show
import matplotlib.pyplot as plt
import xarray as xr
import numpy as np
import pandas as pd
import math
import os
from datetime import datetime
import skimage
import time
import ee
from rasterio.transform import from_bounds

def add_degree_buffer_to_coords(bbox_coords_tuple, buffer_deg):
    """
    Adds a buffer in degrees to a bounding box defined by a tuple of coordinates.
    Returns a new tuple of buffered coordinates.
    """
    minlon, minlat, maxlon, maxlat = bbox_coords_tuple
    return (minlon - buffer_deg, minlat - buffer_deg, maxlon + buffer_deg, maxlat + buffer_deg)


#cloud masking 
#for landsat 4,5,7
def maskL457sr(image):
    qa_mask = image.select('QA_PIXEL').bitwiseAnd(int('11111', 2)).eq(0)
    saturation_mask = image.select('QA_RADSAT').eq(0)
    return image.updateMask(qa_mask).updateMask(saturation_mask).copyProperties(image, ['system:time_start'])
#for landsat 8
def maskL8sr(image):
    qa_mask = image.select('QA_PIXEL').bitwiseAnd(int('11111', 2)).eq(0)
    saturation_mask = image.select('QA_RADSAT').eq(0)
    return image.updateMask(qa_mask).updateMask(saturation_mask).copyProperties(image, ['system:time_start'])


# Rescale bands 
def rescaleL457sr(image):
    # Keep band names intact
    optical = image.select(['SR_B1', 'SR_B2', 'SR_B3', 'SR_B4', 'SR_B5', 'SR_B7']) \
                   .multiply(0.0000275).add(-0.2)
    thermal = image.select('ST_B6').multiply(0.00341802).add(149.0)

    return image.addBands(optical, overwrite=True).addBands(thermal, overwrite=True)
def rescaleL8sr(image):
    # Optical scaling
    optical = image.select(['SR_B2', 'SR_B3', 'SR_B4', 'SR_B5', 'SR_B6', 'SR_B7']) \
                   .multiply(0.0000275).add(-0.2)
    # Thermal scaling
    thermal = image.select('ST_B10').multiply(0.00341802).add(149.0)
    return image.addBands(optical, overwrite=True).addBands(thermal, overwrite=True)


# Band rename for consistancy
#for landsat 4,5,7
def renameL457(image):
    band_names = image.bandNames()

    sr_map = {
        'SR_B1': 'blue',
        'SR_B2': 'green',
        'SR_B3': 'red',
        'SR_B4': 'nir',
        'SR_B5': 'swir1',
        'SR_B7': 'swir2'
    }
    # Only rename existing bands
    available_sr = [k for k in sr_map if band_names.contains(k)]
    renamed = [sr_map[k] for k in available_sr]

    optical = ee.Image(
        ee.Algorithms.If(
            ee.List(available_sr).size().gt(0),
            image.select(available_sr).rename(renamed),
            ee.Image()
        )
    )

    thermal = ee.Image(
        ee.Algorithms.If(
            band_names.contains('ST_B6'),
            image.select('ST_B6').rename(['thermal']),
            ee.Image()
        )
    )
    #define other bands desired to keep
    qa = ee.Image(
        ee.Algorithms.If(
            band_names.contains('QA_PIXEL'),
            image.select('QA_PIXEL'),
            ee.Image()
        )
    )

    return ee.Image.cat([optical, thermal, qa]).copyProperties(image, image.propertyNames())
#for landsat 8
def renameL8sr(image):
    band_names = image.bandNames()

    sr_map = {
        'SR_B2': 'blue',
        'SR_B3': 'green',
        'SR_B4': 'red',
        'SR_B5': 'nir',
        'SR_B6': 'swir1',
        'SR_B7': 'swir2'
    }

    available_sr = [k for k in sr_map if band_names.contains(k)]
    renamed = [sr_map[k] for k in available_sr]

    optical = ee.Image(
        ee.Algorithms.If(
            ee.List(available_sr).size().gt(0),
            image.select(available_sr).rename(renamed),
            ee.Image()
        )
    )

    thermal = ee.Image(
        ee.Algorithms.If(
            band_names.contains('ST_B10'),
            image.select('ST_B10').rename(['thermal']),
            ee.Image()
        )
    )

    qa = ee.Image(
        ee.Algorithms.If(
            band_names.contains('QA_PIXEL'),
            image.select('QA_PIXEL'),
            ee.Image()
        )
    )

    return ee.Image.cat([optical, thermal, qa]).copyProperties(image, image.propertyNames())

# --- Define Index Calculation Functions ---
def add_ndvi(image):
    return image.addBands(image.normalizedDifference(['nir', 'red']).rename('NDVI'))

def add_ndbi(image):
    return image.addBands(image.normalizedDifference(['swir1', 'nir']).rename('NDBI'))

def add_bsi(image):
    # Use expression for BSI for clarity and robustness
    bsi = image.expression(
        '((SWIR1 + RED) - (NIR + BLUE)) / ((SWIR1 + RED) + (NIR + BLUE))',
        {
            'SWIR1': image.select('swir1'),
            'RED': image.select('red'),
            'NIR': image.select('nir'),
            'BLUE': image.select('blue')
        }
    ).rename('BSI')
    return image.addBands(bsi)
    
def add_ndwi(image):
    ndwi = image.normalizedDifference(['green', 'nir']).rename('NDWI')
    return image.addBands(ndwi)


def apply_indices(image):
    image = ee.Image(image) # Ensure input is an ee.Image

    # Check if image has required bands before attempting index calculation
    has_optical_bands = image.bandNames().containsAll(['blue', 'green', 'red', 'nir', 'swir1'])
    
    # Create an initial "indexed" image based on whether optical bands exist
    indexed_image = ee.Algorithms.If(
        has_optical_bands,
        add_ndvi(image),
        image # If no optical bands, return original (un-indexed) image
    )
    indexed_image = ee.Image(indexed_image) # Cast after first index
    
    indexed_image = ee.Algorithms.If(
        has_optical_bands,
        add_ndbi(indexed_image),
        indexed_image
    )
    indexed_image = ee.Image(indexed_image)

    indexed_image = ee.Algorithms.If(
        has_optical_bands,
        add_bsi(indexed_image),
        indexed_image
    )
    indexed_image = ee.Image(indexed_image)

    indexed_image = ee.Algorithms.If(
        has_optical_bands,
        add_ndwi(indexed_image),
        indexed_image
    )
    indexed_image = ee.Image(indexed_image)
    return indexed_image.copyProperties(image, image.propertyNames()) # Copy all properties


# Add LST    
#for landsat 8-4
# from https://www.mdpi.com/2073-4433/16/6/712 
def add_lst(image):
    ndvi = image.select('NDVI')
    bt = image.select('thermal')

    # Dynamic min/max per image
    ndvi_min = ndvi.reduceRegion(
        reducer=ee.Reducer.min(),
        geometry=image.geometry(),
        scale=30,
        maxPixels=1e9
    ).get('NDVI')
    
    ndvi_max = ndvi.reduceRegion(
        reducer=ee.Reducer.max(),
        geometry=image.geometry(),
        scale=30,
        maxPixels=1e9
    ).get('NDVI')
    
    ndvi_min = ee.Number(ndvi_min)
    ndvi_max = ee.Number(ndvi_max)
    # Fraction of vegetation
    fv = ndvi.subtract(ndvi_min).divide(ndvi_max.subtract(ndvi_min)).pow(2).rename('FV')
    # Emissivity
    em = fv.multiply(0.004).add(0.986).rename('EM')
    # Constants
    wavelength = 10.895e-6  # in meters
    rho = 1.438e-2

    # LST expression
    lst = bt.expression(
        'BT / (1 + (lambda * BT / rho) * log(EM)) - 273.15',
        {
            'BT': bt,
            'EM': em,
            'lambda': wavelength,
            'rho': rho
        }
    ).rename('LST')

    return image.addBands([fv, em, lst])


# Export Image collection to netcdf 
def collection_to_array(collection, region, scale, band, max_images):
    collection_list = collection.limit(max_images).toList(max_images)
    arrays = []
    target_shape = None

    for i in range(max_images):
        try:
            img = ee.Image(collection_list.get(i))
            if isinstance(band, list) and len(band) == 1:
                selected_band = band[0] # Extract the band name from the list
            elif isinstance(band, str):
                selected_band = band
            else:
                print(f"Error: Invalid 'band' argument type for image {i}. Expected string or list of one string. Got: {type(band)}")
                continue # Skip this image

            band_img = img.select(selected_band) # Always select a single string band name

            arr = geemap.ee_to_numpy(band_img, region=region, scale=scale)

            if arr is not None and arr.size > 0:
                # Important: After ee_to_numpy, the array should be 2D (height, width).
                # If it's (height, width, 1), we need to squeeze it.
                if arr.ndim == 3 and arr.shape[2] == 1:
                    arr = arr.squeeze(axis=2) # Remove the last dimension if it's 1

                if arr.ndim != 2:
                    print(f"Warning: Array for image {i} (band {selected_band}) is not 2D after processing. Shape: {arr.shape}. Skipping.")
                    continue

                if target_shape is None:
                    target_shape = arr.shape
                    arrays.append(arr)
                else:
                    if arr.shape != target_shape:
                        #print(f"Warning: Image {i} (band {selected_band}) has shape {arr.shape}, expected {target_shape}. Resizing...")
                        resized_arr = np.full(target_shape, np.nan, dtype=arr.dtype)
                        rows_to_copy = min(arr.shape[0], target_shape[0])
                        cols_to_copy = min(arr.shape[1], target_shape[1])
                        resized_arr[:rows_to_copy, :cols_to_copy] = arr[:rows_to_copy, :cols_to_copy]
                        arrays.append(resized_arr)
                    else:
                        arrays.append(arr)
            else:
                print(f"Skipping image {i} (band {selected_band}) because it returned an empty or None array.")

        except Exception as e:
            print(f"Skipping image {i} (band {selected_band}) due to error: {e}")
            continue

    if arrays:
        stacked = np.stack(arrays, axis=0)
        return stacked
    else:
        return None

        
def stack_bands_to_4D(collection, region, scale, band_names, max_images):
    band_arrays = []

    for band in band_names:
        #print(f"Processing band: {band}")
        arr = collection_to_array(collection, region, scale, band, max_images)
        if arr is not None:
            band_arrays.append(arr)
        else:
            print(f"Skipping band {band} due to failed extraction.")

    # Stack along last axis: shape will be (time, rows, cols, bands)
    if band_arrays:
        stacked_4d = np.stack(band_arrays, axis=-1)
        return stacked_4d
    else:
        return None

def save_stack_to_netcdf(stacked_array, band_names, dates, region_bounds, out_path, crs='EPSG:4326'):
    """
    Save a 4D numpy array to NetCDF.
    
    Parameters:
    - stacked_array: numpy array with shape (time, lat, lon, bands)
    - band_names: list of band names (length = bands)
    - dates: list of datetime objects or strings (length = time)
    - region_bounds: tuple (west, south, east, north) in degrees
    - out_path: output netCDF file path
    - crs: coordinate reference system string (default: EPSG:4326)
    """

    time_dim, lat_dim, lon_dim, band_dim = stacked_array.shape

    west, south, east, north = region_bounds
    
    # Generate 1D coordinate arrays for latitude and longitude
    lats = np.linspace(north, south, lat_dim)  # descending order (north to south)
    lons = np.linspace(west, east, lon_dim)
    
    # Convert dates to pandas datetime index if strings
    if isinstance(dates[0], str):
        times = pd.to_datetime(dates)
    else:
        times = pd.to_datetime(dates)

    # Create xarray DataArray with dims and coords
    da = xr.DataArray(
        data=stacked_array,
        dims=["time", "lat", "lon", "band"],
        coords={
            "time": times,
            "lat": lats,
            "lon": lons,
            "band": band_names
        },
        name="LandsatData"
    )
    
    # Add CRS as an attribute
    da.attrs["crs"] = crs

    # Wrap in Dataset for potential multiple variables
    ds = xr.Dataset({"LandsatData": da})

    # Save to NetCDF
    ds.to_netcdf(out_path)
    print(f"Saved NetCDF to {out_path}")

def get_image_dates(collection, max_images=40):
    collection = collection.limit(max_images)
    timestamps = collection.aggregate_array('system:time_start').getInfo()
    dates = [datetime.utcfromtimestamp(ts / 1000).strftime('%Y-%m-%d') for ts in timestamps]
    return dates

def resize_to_match(array, target_shape):
    """Resize a 4D array to match the target shape (time, lat, lon, band)."""
    from skimage.transform import resize
    resized = []
    for i in range(array.shape[0]):
        img = array[i]
        resized_img = resize(img, target_shape[1:], order=1, preserve_range=True, anti_aliasing=True)
        resized.append(resized_img)
    return np.stack(resized, axis=0)
    
def get_chunked_stack(image_collection, region, scale, band_names, chunk_size=40, sleep_time=3):
    """Download the image collection in chunks and stack them."""
    total_images = image_collection.size().getInfo()
    num_chunks = math.ceil(total_images / chunk_size)
    
    all_stacks = []
    all_dates = []

    for i in range(num_chunks):
        print(f"Processing chunk {i+1}/{num_chunks}")
        start = i * chunk_size
        end = min(start + chunk_size, total_images)
        
        chunk = image_collection.toList(chunk_size, start)
        chunk_ic = ee.ImageCollection(chunk)

        try:
            dates_chunk = get_image_dates(chunk_ic)
            stack_chunk = stack_bands_to_4D(chunk_ic, region, scale=scale, band_names=band_names, max_images=chunk_size)
            all_stacks.append(stack_chunk)
            all_dates.extend(dates_chunk)
        except Exception as e:
            print(f"Chunk {i+1} failed: {e}")
        
        time.sleep(sleep_time)  # respect GEE quotas
    
            # Get the target shape from the first stack
        target_shape = all_stacks[0].shape
        
        # Resize each to the same shape
        resized_stacks = [resize_to_match(a, target_shape) if a.shape != target_shape else a for a in all_stacks]
        
        # Concatenate safely
        stacked_final = np.concatenate(resized_stacks, axis=0)
    return stacked_final, all_dates

#or to geo tiff for each band 


def to_geotiff_stack(stacked_array, region_bounds, band_names, dates, out_dir, crs='EPSG:4326'):
    """
    Writes one GeoTIFF file per band, where each file contains time as the band dimension.
    
    Parameters:
        stacked_array: 4D np.array (time, lat, lon, bands)
        region_bounds: (west, south, east, north)
        band_names: list of band names
        dates: list of datetime or str (used for band descriptions)
        out_dir: folder to save TIFFs
        crs: Coordinate reference system
    """
    time_dim, height, width, num_bands = stacked_array.shape
    transform = from_bounds(*region_bounds, width=width, height=height)
    os.makedirs(out_dir, exist_ok=True)

    for b in range(num_bands):
        band_array = stacked_array[:, :, :, b]  # shape: (time, lat, lon)
        file_path = os.path.join(out_dir, f"{band_names[b]}.tif")

        with rasterio.open(
            file_path,
            'w',
            driver='GTiff',
            height=height,
            width=width,
            count=time_dim,
            dtype=stacked_array.dtype,
            crs=crs,
            transform=transform,
        ) as dst:
            for t in range(time_dim):
                dst.write(band_array[t, :, :], t + 1)
                if dates:
                    date_str = dates[t] if isinstance(dates[t], str) else dates[t].strftime('%Y-%m-%d')
                    dst.set_band_description(t + 1, date_str)

        print(f"Saved {file_path}")



