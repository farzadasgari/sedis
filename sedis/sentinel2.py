import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from config import *
import ee
import requests
import datetime
import rasterio
import numpy as np
import pandas as pd

ee.Authenticate()
ee.Initialize(project=project_id)

start_date = '2017-03-28'
end_date = '2025-10-12'

aoi = ee.Geometry.Rectangle([lon_min, lat_min, lon_max, lat_max], 'EPSG:4326', False)

dataset = ee.ImageCollection(collection)

filtered = dataset.filterDate(start_date, end_date) \
                  .filterBounds(aoi) \
                  .filter(ee.Filter.lt('CLOUDY_PIXEL_PERCENTAGE', 20))

def mask_clouds(image):
    scl = image.select('SCL')
    cloud_shadow_snow_mask = scl.eq(8).Or(scl.eq(9)).Or(scl.eq(10)).Or(scl.eq(3)).Or(scl.eq(11))
    return image.updateMask(cloud_shadow_snow_mask.Not())

def mask_water(image):
    scl = image.select('SCL')
    water_mask = scl.eq(6)
    return image.updateMask(water_mask)

selected = filtered.select(bands + ['SCL'])

masked = selected.map(mask_clouds).map(mask_water)

def mosaic_by_date(col):
    dates = col.toList(col.size()).map(lambda img: ee.Image(img).date().format('YYYY-MM-dd')).distinct()
    
    def create_mosaic(date_str):
        date = ee.Date(date_str)
        day_imgs = col.filterDate(date, date.advance(1, 'day')).mosaic()
        return day_imgs.set('Date', date.format('YYYY-MM-dd'))
    
    return ee.ImageCollection(dates.map(create_mosaic))

mosaicked = mosaic_by_date(masked)

def compute_means(image):
    means = image.reduceRegion(
        reducer=ee.Reducer.mean(),
        geometry=aoi,
        scale=10,
        bestEffort=True
    )
    date = image.get('Date')
    return ee.Feature(None, means).set('Date', date)

means_collection = ee.FeatureCollection(mosaicked.map(compute_means))

download_url = means_collection.getDownloadURL('CSV')

temp_csv_file = 'temp_kansas.csv'

response = requests.get(download_url)
with open(temp_csv_file, 'wb') as f:
    f.write(response.content)

df = pd.read_csv(temp_csv_file)

band_order = ['B1', 'B2', 'B3', 'B4', 'B5', 'B6', 'B7', 'B8', 'B8A', 'B9', 'B11', 'B12', 'AOT']

available_bands = [col for col in df.columns if col in band_order]
sorted_bands = sorted(available_bands, key=lambda x: band_order.index(x) if x in band_order else len(band_order))

columns_to_remove = ['system:index', 'SCL', '.geo']
columns_to_keep = ['Date'] + sorted_bands

other_columns = [col for col in df.columns 
                if col not in columns_to_remove 
                and col not in columns_to_keep]

columns_to_keep.extend(other_columns)

df_clean = df[columns_to_keep]

csv_file = 'dataset/sentinel.csv'
df_clean.to_csv(csv_file, index=False)

os.remove(temp_csv_file)

print(f"Download and processing successful! Cleaned CSV saved")
