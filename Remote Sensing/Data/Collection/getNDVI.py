import ee
import os
import geemap
import numpy as np
from PIL import Image

# Initialize Earth Engine
ee.Initialize(project='ee-sciencefair2425')

def get_ndvi(image):
    """Calculate NDVI from NAIP imagery."""
    ndvi = image.normalizedDifference(['N', 'R']).rename('NDVI')  # N is Near Infrared, R is Red
    return image.addBands(ndvi)

def read_coordinates(file_path):
    """Read coordinates from a text file and return as a list of tuples."""
    with open(file_path, 'r') as f:
        lines = f.readlines()
    coordinates = []
    for line in lines:
        lat, lon = map(float, line.strip().split(','))
        coordinates.append((lat, lon))
    return coordinates

def create_aoi(lat, lon):
    """Create a 1x1 mile square (0.5-mile buffer around the point)."""
    center_point = ee.Geometry.Point([lon, lat])
    region = center_point.buffer(804.672).bounds()  # 804.672 meters = 0.5 miles
    return region

def apply_color_map(ndvi_array):
    """Apply a gradient color map to the NDVI array for better visualization."""
    # Define color gradient for NDVI values (from low to high)
    color_palette = [
        (255, 0, 0),   # Low NDVI (water)
        (255, 0, 0),  # Bare soil
        (255, 255, 0),   # Sparse vegetation (yellow)
        (0, 255, 0)      # Dense vegetation (green)
    ]

    # Create an empty RGB image
    rgb_image = np.zeros((ndvi_array.shape[0], ndvi_array.shape[1], 3), dtype=np.uint8)

    # Normalize NDVI values to [0, 1]
    ndvi_normalized = np.clip((ndvi_array + 1) / 2, 0, 1)  # Normalize NDVI from [-1, 1] to [0, 1]

    # Interpolate colors based on normalized NDVI values
    for i in range(len(color_palette) - 1):
        lower_color = np.array(color_palette[i])
        upper_color = np.array(color_palette[i + 1])
        
        # Create a mask for the range of normalized NDVI values
        mask = (ndvi_normalized >= i / (len(color_palette) - 1)) & (ndvi_normalized <= (i + 1) / (len(color_palette) - 1))
        
        # Interpolate between the two colors
        ratio = (ndvi_normalized[mask] * (len(color_palette) - 1) - i).reshape(-1, 1)
        rgb_image[mask] = lower_color + ratio * (upper_color - lower_color)

    return rgb_image

def process_directories(root_dir, coord_file):
    """Process subdirectories and fetch NDVI for each coordinate."""
    coordinates = read_coordinates(coord_file)
    
    subfolders = next(os.walk(root_dir))[1]
    subfolders = sorted(subfolders, key=lambda x: int(x))
    
    for idx, subfolder in enumerate(subfolders):
        if idx >= len(coordinates):  # Ensure we don't exceed the number of coordinates
            print(f"No coordinate available for subfolder: {subfolder}. Skipping.")
            continue
        
        lat, lon = coordinates[idx]
        aoi = create_aoi(lat, lon)  # Create the area of interest (1x1 mile square)
        
        collection = (ee.ImageCollection('USDA/NAIP/DOQQ')
                      .filterBounds(aoi)
                      .filterDate('2018-01-01', '2022-12-31')  # Adjust date range as needed
                      .map(get_ndvi))
        
        # Get the median NDVI image and clip it to the AOI
        ndvi_image = collection.select('NDVI').median().clip(aoi)
        
        # Export the NDVI GeoTIFF with original values
        output_path_tif = os.path.join(root_dir, subfolder, 'ndvi.tif')
        
        print(f"Exporting NDVI GeoTIFF for {subfolder} (Lat: {lat}, Lon: {lon}) to {output_path_tif}...")
        geemap.ee_export_image(ndvi_image, filename=output_path_tif, scale=10, region=aoi)
        
        print(f"Exported NDVI GeoTIFF to: {output_path_tif}")

        # Convert TIFF to PNG using cached version
        print(f"Converting NDVI TIFF to colored PNG for {subfolder}...")
        
        # Read the TIFF file using PIL and convert it to an array
        ndvi_array = np.array(Image.open(output_path_tif))

        # Apply color mapping to create an RGB image from the NDVI array
        colored_ndvi = apply_color_map(ndvi_array)

        output_path_png = os.path.join(root_dir, subfolder, 'ndvi.png')
        
        # Save the colored PNG image
        Image.fromarray(colored_ndvi).save(output_path_png)
        
        print(f"Exported colored NDVI PNG to: {output_path_png}")

# Specify paths
root_directory = 'Datasets/'  # Root directory containing subdirectories
coordinates_file = 'Collection/coordinates.txt'  # Path to the coordinates.txt file

# Process all subdirectories and fetch enhanced NDVI data
process_directories(root_directory, coordinates_file)
