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

ee.Authenticate()
ee.Initialize()

#name of output
out_path = "jd_landsat.nc"

#define coordinates to use 
coords_jd = (39.0053612, 21.4210088, 39.3253612, 21.7410088)  # Jeddah
coords_md = (39.4511216, 24.311153, 39.7711216, 24.631153)   # Medina
coords_mk = (39.666869, 21.260847, 39.986869, 21.580847)   # Mecca
coords_tf = (40.2558308, 21.1102801, 40.5758308, 21.4302801)   # Taif


buffered_coords = add_degree_buffer_to_coords(coords_jd, 0.3)
bbox_jd = ee.Geometry.BBox(*buffered_coords) 
bbox = bbox_jd


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
WRS_PATH = 170
WRS_ROW = 45
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
exclude_ids = ['1_1_2_LT05_170045_19871122', '1_1_2_LT05_170045_19881226', '2_LC08_170045_20160801']
landsat_indexed = landsat_indexed.filter(ee.Filter.inList('system:index', exclude_ids).Not())

#Export to NetCDF 
bands = ee.Image(landsat_indexed.first()).bandNames().getInfo()
stack, dates = get_chunked_stack(landsat_indexed, bbox, 1000, bands, sleep_time=10) # here 1 km becuase 30 m is too small 
save_stack_to_netcdf(stack, bands, dates, coords_jd, out_path)
