import os
import requests
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import pystac_client
import planetary_computer


# Parse command-line argument
parser = argparse.ArgumentParser()
parser.add_argument("--num-workers", type=int, default=8, help="Number of parallel threads")
args = parser.parse_args()

# Setup
planetary_endpoint = "https://planetarycomputer.microsoft.com/api/stac/v1/"
bbox = [38.837314, 21.092348, 39.592624, 22.136691]  # Jeddah
product = ['landsat-c2-l2']
date_range = "2022-11-06" # '1982-08-22/2025-06-15'
bands_to_download = ['lwir11', 'qa_radsat']
output_folder = "/mnt/datawaha/hyex/gahwagrw/SaudiUHI/sat_data/raw/Jeddah/landsat"
os.makedirs(output_folder, exist_ok=True)

# Search the STAC Catalog
catalog = pystac_client.Client.open(planetary_endpoint)
item_collection = catalog.search(
    collections=product,
    bbox=bbox,
    datetime=date_range,
).item_collection()

# Define download function
def download_band(item, band):
    try:
        item = planetary_computer.sign(item)
        scene_id = item.id
        scene_date = (
                        signed_item.datetime.date()
                        if signed_item.datetime
                        else signed_item.properties.get('start_datetime', 'unknown')
                    )


        if band not in item.assets:
            return f"[{scene_id}] Band {band} not found, skipping."

        asset = item.assets[band]
        url = asset.href
        filename = f"{scene_id}_{band}.tif"
        filepath = os.path.join(output_folder, filename)

        if os.path.exists(filepath):
            return f"[{scene_id}] {band} already exists."

        with requests.get(url, stream=True) as r:
            r.raise_for_status()
            with open(filepath, 'wb') as f:
                for chunk in r.iter_content(chunk_size=8192):
                    f.write(chunk)

        return f"Downloaded {filename} from {scene_date}"

    except Exception as e:
        return f"Error downloading {item.id} {band}: {e}"

# Prepare download tasks
tasks = []
for item in item_collection:
    for band in bands_to_download:
        tasks.append((item, band))

# Run in parallel
print(f"Starting parallel download with {args.num_workers} threads for {len(tasks)} files...")

with ThreadPoolExecutor(max_workers=args.num_workers) as executor:
    futures = [executor.submit(download_band, item, band) for item, band in tasks]

    for future in as_completed(futures):
        print(future.result())
