import os
import requests
import planetary_computer
import pystac_client
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor
from tqdm import tqdm

# -------------------- Configuration --------------------
planetary_endpoint = "https://planetarycomputer.microsoft.com/api/stac/v1/"
product = 'modis-11A2-061'  

# Area of interest: Jeddah (modify as needed)
bbox = [38.837314, 21.092348, 39.592624, 22.136691]

# Time range: 2015 to mid-2025 (use only what's available)
date_range = '1982-08-22/2025-12-31'

# Bands you want to download
bands_to_download = [
    "LST_Day_1km",
    "LST_Night_1km",
    "QC_Day",
    "QC_Night",
    "Day_view_time",
    "Night_view_time",
    'Emis_31', 
    'Emis_32'
    'Clear_sky_days'
]

# Output path
output_folder = "/mnt/datawaha/hyex/gahwagrw/SaudiUHI/sat_data/raw/Jeddah/modis"
os.makedirs(output_folder, exist_ok=True)

# -------------------- Fetch Items --------------------
catalog = pystac_client.Client.open(planetary_endpoint)
search = catalog.search(
    collections=[product],
    bbox=bbox,
    datetime=date_range
)
items = list(search.get_all_items())
print(f"Found {len(items)} MODIS Aqua scenes")

# -------------------- Helper Functions --------------------
def get_scene_date(item):
    """Extract scene start date from STAC metadata."""
    start_dt = item.properties.get("start_datetime")
    if start_dt is None:
        print(f"[WARNING] No start_datetime for item {item.id}. Skipping.")
        return None
    return datetime.fromisoformat(start_dt.replace("Z", "")).date()

def download_asset(item, band):
    """Download a specific band for a single STAC item."""
    try:
        signed_item = planetary_computer.sign(item)
        scene_id = signed_item.id
        scene_date = get_scene_date(signed_item)
        if scene_date is None:
            return f"[{scene_id}] Skipped (no date)"

        if band not in signed_item.assets:
            return f"[{scene_id}] Band '{band}' not available"

        asset = signed_item.assets[band]
        url = asset.href
        filename = f"{scene_date}_{scene_id}_{band}.tif"
        filepath = os.path.join(output_folder, filename)

        if os.path.exists(filepath):
            return f"[{scene_id}] {band} already exists"

        with requests.get(url, stream=True, timeout=60) as r:
            r.raise_for_status()
            with open(filepath, "wb") as f:
                for chunk in r.iter_content(chunk_size=8192):
                    f.write(chunk)

        return f"[{scene_id}] {band} downloaded"

    except Exception as e:
        return f"[{item.id}] {band} failed: {e}"

# -------------------- Parallel Download Function --------------------
def download_all(items, bands, max_workers=32):
    jobs = [(item, band) for item in items for band in bands]
    print(f" Launching download of {len(jobs)} files with {max_workers} workers")

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        results = list(tqdm(
            executor.map(lambda job: download_asset(*job), jobs),
            total=len(jobs)
        ))

    print("\n Download Summary:")
    for res in results:
        print(res)

# -------------------- Execute --------------------
download_all(items, bands_to_download, max_workers=32)
