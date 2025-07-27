import requests
import os
import pystac_client
import planetary_computer

# define varibles 
planetary_endpoint = "https://planetarycomputer.microsoft.com/api/stac/v1/"

bbox = [38.837314, 21.092348, 39.592624, 22.136691]
product = ['landsat-c2-l2']
date_range = '1982-08-22/2013-03-28'
bands_to_download = ['lwir'] 

output_folder = "/mnt/datawaha/hyex/gahwagrw/SaudiUHI/sat_data/raw/Jeddah/landsat"


#get data 
catalog = pystac_client.Client.open(planetary_endpoint)

item_collection = catalog.search(
    collections=product,
    bbox= bbox,
    datetime=date_range,
).item_collection()

num_files = len(item_collection)

#download band 
for item in item_collection:
    item = planetary_computer.sign(item)
    scene_id = item.id
    scene_date = item.datetime.date()

    for band in bands_to_download:
        if band not in item.assets:
            print(f"[{scene_id}] Band {band} not found, skipping.")
            continue

        asset = item.assets[band]
        url = asset.href
        filename = f"{scene_id}_{band}.tif"
        filepath = os.path.join(output_folder, filename)

        if os.path.exists(filepath):
            print(f"[{scene_id}] {band} already downloaded.")
            continue

        print(f"Downloading {filename} from {scene_date}...")
        try:
            with requests.get(url, stream=True) as r:
                r.raise_for_status()
                with open(filepath, 'wb') as f:
                    for chunk in r.iter_content(chunk_size=8192):
                        f.write(chunk)
        except Exception as e:
            print(f"Failed to download {filename}: {e}")

