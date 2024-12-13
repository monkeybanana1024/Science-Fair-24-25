import ee
import os
import geemap

# Initialize Earth Engine
ee.Initialize(project='ee-sciencefair2425')

def get_ndvi(image):
    """Calculate NDVI from Sentinel-2 imagery."""
    ndvi = image.normalizedDifference(['B8', 'B4']).rename('NDVI')
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

def remap_ndvi(ndvi_image):
    """Remap positive NDVI values to [0, 255] while leaving negative values unchanged."""
    ndvi_remapped = ndvi_image.where(ndvi_image.gt(0), 
                                      ndvi_image.subtract(-1).divide(2).multiply(255).byte())
    return ndvi_remapped

def enhance_image(ndvi_image):
    """Enhance NDVI image using contrast stretching."""
    min_max = ndvi_image.reduceRegion(
        reducer=ee.Reducer.minMax(),
        geometry=ndvi_image.geometry(),
        scale=30,
        bestEffort=True
    )

    min_val = ee.Number(min_max.get('NDVI_min'))
    max_val = ee.Number(min_max.get('NDVI_max'))

    # Apply contrast stretching
    stretched_ndvi = ndvi_image.subtract(min_val).divide(max_val.subtract(min_val)).multiply(255).byte()

    return stretched_ndvi

def apply_color_map(ndvi_image):
    """Apply a color map to the NDVI image for better visualization."""
    # Define color palette for NDVI values
    color_palette = [
        'white',   # Low NDVI (water)
        'white',  # Bare soil
        'yellow', # Sparse vegetation
        'green'   # Dense vegetation
    ]

    # Create a visualization dictionary
    vis_params = {
        'min': 0,
        'max': 255,
        'palette': color_palette
    }

    return ndvi_image.visualize(**vis_params)

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
        
        collection = (ee.ImageCollection('COPERNICUS/S2_SR')
                      .filterBounds(aoi)
                      .filterDate('2022-01-01', '2022-12-31')
                      .map(get_ndvi))
        
        # Get the median NDVI image and clip it to the AOI
        ndvi_image = collection.select('NDVI').median().clip(aoi)
        
        # Remap and enhance NDVI values
        ndvi_remapped = remap_ndvi(ndvi_image)
        enhanced_ndvi = enhance_image(ndvi_remapped)

        # Export the NDVI GeoTIFF without color mapping
        output_path_tif = os.path.join(root_dir, subfolder, 'ndvi.tif')
        
        print(f"Exporting NDVI GeoTIFF for {subfolder} (Lat: {lat}, Lon: {lon}) to {output_path_tif}...")
        geemap.ee_export_image(ndvi_image, filename=output_path_tif, scale=10, region=aoi)
        
        print(f"Exported NDVI GeoTIFF to: {output_path_tif}")

        # Normalize and apply color mapping for PNG output
        colored_ndvi = apply_color_map(enhanced_ndvi)
        
        output_path_png = os.path.join(root_dir, subfolder, 'ndvi.png')
        
        print(f"Exporting colored NDVI PNG for {subfolder} to {output_path_png}...")
        geemap.ee_export_image(colored_ndvi, filename=output_path_png, scale=10, region=aoi)
        
        print(f"Exported colored NDVI PNG to: {output_path_png}")

# Specify paths
root_directory = 'Datasets/'  # Root directory containing subdirectories
coordinates_file = 'Collection/coordinates.txt'  # Path to the coordinates.txt file

# Process all subdirectories and fetch enhanced NDVI data
process_directories(root_directory, coordinates_file)
