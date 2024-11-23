import ee
import pysda
import numpy as np

# Initialize Earth Engine
ee.Initialize(project="ee-sciencefair2425")

def get_soil_and_satellite_data(lat, lon):
    # Create a point geometry
    point = ee.Geometry.Point([lon, lat])

    # Get soil data using pySDA
    soil_data = pysda.soil_survey_area(lat, lon)

    # Get Landsat 8 image
    landsat = ee.ImageCollection('LANDSAT/LC08/C02/T1_L2') \
        .filterBounds(point) \
        .sort('CLOUD_COVER') \
        .first()

    # Calculate NDVI (Normalized Difference Vegetation Index)
    nir = landsat.select('SR_B5')
    red = landsat.select('SR_B4')
    ndvi = nir.subtract(red).divide(nir.add(red)).rename('NDVI')

    # Get average rainfall data (using CHIRPS dataset)
    rainfall = ee.ImageCollection('UCSB-CHG/CHIRPS/DAILY') \
        .filterDate(ee.Date.now().advance(-1, 'year'), ee.Date.now()) \
        .mean() \
        .select('precipitation')

    # Extract values
    ndvi_value = ndvi.reduceRegion(ee.Reducer.mean(), point, 30).get('NDVI').getInfo()
    rainfall_value = rainfall.reduceRegion(ee.Reducer.mean(), point, 5000).get('precipitation').getInfo()

    return {
        'soil_data': soil_data,
        'ndvi': ndvi_value,
        'average_annual_rainfall_mm': rainfall_value * 365  # Convert daily to annual
    }

# Example usage
lat, lon = 40.7128, -74.0060  # New York City coordinates
result = get_soil_and_satellite_data(lat, lon)

print("Soil Data:", result['soil_data'])
print("NDVI (Vegetation Index):", result['ndvi'])
print("Average Annual Rainfall (mm):", result['average_annual_rainfall_mm'])