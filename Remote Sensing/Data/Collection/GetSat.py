import ee
import os
import geemap
import requests

# Initialize Earth Engine
ee.Initialize(project='ee-sciencefair2425')

def read_coordinates(file_path):
    """Read coordinates from a text file and return as a list of tuples."""
    with open(file_path, 'r') as f:
        return [tuple(map(float, line.strip().split(','))) for line in f]

def create_aoi(lat, lon):
    """Create a 1x1 mile square (0.5-mile buffer around the point)."""
    center_point = ee.Geometry.Point([lon, lat])
    return center_point.buffer(804.672).bounds()

def process_directories(root_dir, coord_file):
    """Process subdirectories and fetch NAIP imagery for each coordinate."""
    coordinates = read_coordinates(coord_file)
    subfolders = sorted([f for f in os.listdir(root_dir) if os.path.isdir(os.path.join(root_dir, f))], key=int)
    
    for idx, subfolder in enumerate(subfolders):
        if idx >= len(coordinates):
            print(f"No coordinate available for subfolder: {subfolder}. Skipping.")
            continue
        
        lat, lon = coordinates[idx]
        aoi = create_aoi(lat, lon)

        # Try to get NAIP imagery from the current year down to 6 years prior
        found_image = False
        for year in range(2022, 2015, -1):  # Adjust starting year as needed
            naip = (ee.ImageCollection('USDA/NAIP/DOQQ')
                    .filterBounds(aoi)
                    .filterDate(f'{year}-01-01', f'{year}-12-31')
                    .sort('system:time_start', False))
            
            # Check if there are any images in the collection
            if naip.size().getInfo() > 0:
                found_image = True
                # Create a mosaic of the images for the selected year
                naip_image = naip.mosaic()
                break
        
        if not found_image:
            print(f"No NAIP imagery available for {subfolder} (Lat: {lat}, Lon: {lon}) within the last 6 years. Skipping.")
            continue
        
        rgb_image = naip_image.select(['R', 'G', 'B'])
        
        # Set visualization parameters
        vizParams = {'bands': ['R', 'G', 'B'], 'min': 0, 'max': 255}
        rgb_image_vis = rgb_image.visualize(**vizParams)
        rgb_image_clipped = rgb_image_vis.clip(aoi)
        
        output_path_png = os.path.join(root_dir, subfolder, 'sat.png')
        
        print(f"Exporting NAIP RGB image for {subfolder} (Lat: {lat}, Lon: {lon}) to {output_path_png}...")
        
        url = rgb_image_clipped.getThumbURL({'dimensions': 1024, 'format': 'png', 'region': aoi})

        response = requests.get(url)
        if response.status_code == 200:
            with open(output_path_png, 'wb') as f:
                f.write(response.content)
            print(f"Exported NAIP RGB image to: {output_path_png}")
        else:
            print(f"Failed to download image. Status code: {response.status_code}")

# Specify paths
root_directory = 'Datasets/'  # Root directory containing subdirectories
coordinates_file = 'Collection/coordinates.txt'  # Path to the coordinates.txt file

# Process all subdirectories and fetch NAIP imagery
process_directories(root_directory, coordinates_file)
