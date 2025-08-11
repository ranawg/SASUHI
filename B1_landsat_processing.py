import geetools
import ee
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
from A1_landsat_functions import *
import cftime

ee.Authenticate()
ee.Initialize()

#name of output
out_path = "rd_landsat2.nc"

#define coordinates to use 
coords_jd = (39.0053612, 21.4210088, 39.3253612, 21.7410088)  # Jeddah
coords_md = (39.4511216, 24.311153, 39.7711216, 24.631153)   # Medina
coords_mk = (39.698,21.078,40.222,21.694)     # Mecca
coords_tf = (40.2558308, 21.1102801, 40.5758308, 21.4302801)   # Taif
#coords_rd # Riyadh
coord_rd = [[[46.4328690955,24.1659202907],[46.6345003059,25.4285645773],[47.5182109888,25.2818982646],
             [47.5209967609,24.1956804525],[46.4328690955,24.1659202907]]] #riyad polygon

coords = coords_mk

buffered_coords = add_degree_buffer_to_coords(coords, 0.005)
bbox = ee.Geometry.BBox(*buffered_coords) 
#for polygon 
bbox = ee.Geometry.Polygon(coord_rd)

#get Data: Landsat 
# Landsat collections (T1 = Tier 1 Surface Reflectance where available)
landsat1 = ee.ImageCollection("LANDSAT/LM01/C02/T1")
landsat2 = ee.ImageCollection("LANDSAT/LM02/C02/T1")
landsat3 = ee.ImageCollection("LANDSAT/LM03/C02/T1")
landsat4 = ee.ImageCollection("LANDSAT/LT04/C02/T1_L2")
landsat5 = ee.ImageCollection("LANDSAT/LT05/C02/T1_L2")
landsat7 = ee.ImageCollection("LANDSAT/LE07/C02/T1_L2")
landsat8 = ee.ImageCollection("LANDSAT/LC08/C02/T1_L2")

# Merge Landsat collections based on sensors
landsat_1_3 = landsat1.merge(landsat2).merge(landsat3).filterBounds(bbox).sort("DATE_ACQUIRED")
landsat_4_7 = landsat4.merge(landsat5).merge(landsat7).filterBounds(bbox).sort("DATE_ACQUIRED")
landsat8 = landsat8.filterBounds(bbox).sort("DATE_ACQUIRED")

# Define panel to use 
# Jeddah: Path: 170 Row: 045 
#makkah 169, 45
 #riyadh 165, 43
WRS_PATH = 165
WRS_ROW = 43
landsat_1_3 = landsat_1_3.filter(ee.Filter.And(ee.Filter.eq('WRS_PATH', WRS_PATH),ee.Filter.eq('WRS_ROW', WRS_ROW))).map(lambda img: img.clip(bbox))
landsat_4_7 = landsat_4_7.filter(ee.Filter.And(ee.Filter.eq('WRS_PATH', WRS_PATH),ee.Filter.eq('WRS_ROW', WRS_ROW))).map(lambda img: img.clip(bbox))
landsat8    = landsat8.filter(ee.Filter.And(ee.Filter.eq('WRS_PATH', WRS_PATH),ee.Filter.eq('WRS_ROW', WRS_ROW))).map(lambda img: img.clip(bbox))

#apply processing for cloud mask and rename bands
landsat_4_7 = landsat_4_7.map(maskL457sr).map(rescaleL457sr).map(renameL457)
landsat8 = landsat8.map(maskL8sr).map(rescaleL8sr).map(renameL8sr)
#merge landsat data sets 
landsat = landsat_4_7.merge(landsat8)

#apply diffent indices 
landsat_indexed = landsat.map(apply_indices)

#Add LST
landsat_indexed = landsat_indexed.map(add_lst)

#exlude dates where there are null and keep getting errors
exclude_ids = ['1_1_2_LT05_170045_19871122', '1_1_2_LT05_170045_19881226', '2_LC08_170045_20160801', '1_1_2_LT05_169045_19960312', '2_LC08_169045_20200720', '1_1_2_LT05_165043_19850505', '1_1_2_LT05_165043_19850505', '1_1_2_LT05_165043_19930103', '1_1_2_LT05_165043_19971130',
              '1_2_LE07_165043_20190511', '1_2_LE07_165043_20190511']
landsat_indexed = landsat_indexed.filter(ee.Filter.inList('system:index', exclude_ids).Not())


#run 2 to recover the rest of the dates by exluding already run dates 
ds = xr.open_dataset("/mnt/datawaha/hyex/gahwagrw/SaudiUHI/SASUHI/rd_landsat.nc")
ds['time'] = pd.to_datetime(ds['time'].values)
last_date = ds['time'].max().item()
last_date_str = pd.to_datetime(last_date).strftime('%Y-%m-%d')
last_processed_date = ee.Date(last_date_str)
landsat_indexed = landsat_indexed.filterDate(last_processed_date.advance(1, 'day'), ee.Date('2025-12-31'))


#Export to NetCDF 
bands = ee.Image(landsat_indexed.first()).bandNames().getInfo()
stack, dates = get_chunked_stack(landsat_indexed, bbox, 1000, bands, sleep_time=10) # here 1 km becuase 30 m is too small 
save_stack_to_netcdf(stack, bands, dates, coords, out_path)



