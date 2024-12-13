import os
import numpy as np
from osgeo import gdal
import matplotlib.pyplot as plt

def calculate_slope_and_aspect(input_dem_path, output_slope_path, output_aspect_path):
    """Calculate slope and aspect from a DEM using GDAL."""
    cmd_slope = f"gdaldem slope -compute_edges {input_dem_path} {output_slope_path}"
    cmd_aspect = f"gdaldem aspect -compute_edges {input_dem_path} {output_aspect_path}"
    os.system(cmd_slope)
    os.system(cmd_aspect)

def create_directional_slope(slope_file, aspect_file, output_file):
    """Create an 8-band directional slope raster."""
    slope_ds = gdal.Open(slope_file)
    aspect_ds = gdal.Open(aspect_file)
    
    slope = slope_ds.GetRasterBand(1).ReadAsArray()
    aspect = aspect_ds.GetRasterBand(1).ReadAsArray()
    
    # Convert aspect to radians
    aspect_rad = np.deg2rad(aspect)
    
    # Create 8 bands for N, NE, E, SE, S, SW, W, NW
    bands = []
    for i in range(8):
        angle = i * 45  # 0, 45, 90, 135, 180, 225, 270, 315 degrees
        weight = np.cos(aspect_rad - np.deg2rad(angle))
        directional_slope = slope * weight
        bands.append(directional_slope)
    
    # Normalize each band to [0, 255]
    bands_normalized = [(band - np.min(band)) / (np.max(band) - np.min(band)) * 255 for band in bands]
    
    # Create a new 8-band raster
    driver = gdal.GetDriverByName('GTiff')
    out_ds = driver.Create(output_file, slope_ds.RasterXSize, slope_ds.RasterYSize, 8, gdal.GDT_Byte)
    
    # Copy geotransform and projection from the original slope file
    out_ds.SetGeoTransform(slope_ds.GetGeoTransform())
    out_ds.SetProjection(slope_ds.GetProjection())
    
    for i, band in enumerate(bands_normalized):
        out_band = out_ds.GetRasterBand(i + 1)  # GDAL bands are 1-indexed
        out_band.WriteArray(band.astype(np.uint8))  # Save as uint8
    
    out_ds = None  # Close the dataset
    slope_ds = None
    aspect_ds = None

def colorize_slope(slope_band):
    """Colorize a single slope band based on its values."""
    cmap = plt.get_cmap('RdYlGn')  # Green to Red colormap
    normalized_band = (slope_band - np.min(slope_band)) / (np.max(slope_band) - np.min(slope_band))
    colored_image = cmap(normalized_band)[:, :, :3]  # Get RGB channels
    return (colored_image * 255).astype(np.uint8)  # Convert to uint8

def save_colorized_slope(slope_data, output_dir):
    """Save colorized slope images for each direction."""
    directions = ['N', 'NE', 'E', 'SE', 'S', 'SW', 'W', 'NW']
    
    for i in range(slope_data.shape[2]):  # Assuming shape is (height, width, num_bands)
        colored_image = colorize_slope(slope_data[:, :, i])
        
        # Save as PNG
        output_filename = os.path.join(output_dir, f'{directions[i]}.png')
        plt.imsave(output_filename, colored_image)
        print(f"Saved colorized slope image: {output_filename}")

def process_slope(input_dem_path):
    """Process the DEM to calculate and colorize slopes."""
    # Create output directory for slopes
    output_dir = os.path.join(os.path.dirname(input_dem_path), 'slope')
    os.makedirs(output_dir, exist_ok=True)

    # Calculate slope and aspect
    output_slope_path = os.path.join(output_dir, 'slope.tif')
    output_aspect_path = os.path.join(output_dir, 'aspect.tif')
    calculate_slope_and_aspect(input_dem_path, output_slope_path, output_aspect_path)

    # Create directional slope (normalized to [0-255])
    output_directional_slope_path = os.path.join(output_dir, 'slope.tif')
    create_directional_slope(output_slope_path, output_aspect_path, output_directional_slope_path)

    # Load the directional slope GeoTIFF
    with gdal.Open(output_directional_slope_path) as src:
        bands = src.ReadAsArray()  # Shape: (num_bands, height, width)
        bands = bands.transpose(1, 2, 0)  # Reshape to (height, width, num_bands)

        # Save colorized versions of slopes
        save_colorized_slope(bands.astype(np.float32), output_dir)

def colorize_aspect(aspect_file, output_png_path):
    """Colorize the aspect raster using a circular colormap."""
    # Open the aspect raster
    aspect_ds = gdal.Open(aspect_file)
    aspect = aspect_ds.GetRasterBand(1).ReadAsArray()

    # Normalize aspect values to [0, 1] for colormap mapping
    # Aspect values range from 0 to 360; flat areas are often -1
    normalized_aspect = np.where(aspect >= 0, aspect / 360.0, np.nan)

    # Create a colormap (HSV: circular gradient)
    cmap = plt.cm.hsv  # HSV colormap for circular direction mapping

    # Apply the colormap
    colored_aspect = cmap(normalized_aspect)[:, :, :3]  # Get RGB channels

    # Replace NaN values (flat areas) with gray (e.g., [128, 128, 128])
    flat_color = [0.5, 0.5, 0.5]  # Gray in normalized RGB
    nan_mask = np.isnan(normalized_aspect)
    colored_aspect[nan_mask] = flat_color

    # Convert to uint8 for saving as an image
    colored_aspect_uint8 = (colored_aspect * 255).astype(np.uint8)

    # Save as PNG
    plt.imsave(output_png_path, colored_aspect_uint8)
    print(f"Saved colorized aspect image: {output_png_path}")

def process_slope_and_aspect(input_dem_path):
    """Process DEM to calculate slope and aspect, and colorize both."""
    output_dir = os.path.join(os.path.dirname(input_dem_path), 'slope')
    os.makedirs(output_dir, exist_ok=True)

    output_slope_path = os.path.join(output_dir, 'slope.tif')
    output_aspect_path = os.path.join(output_dir, 'aspect.tif')
    
    # Calculate slope and aspect
    calculate_slope_and_aspect(input_dem_path, output_slope_path, output_aspect_path)

    # Create directional slope (normalized to [0–255])
    output_directional_slope_path = os.path.join(output_dir, 'slope.tif')
    create_directional_slope(output_slope_path, output_aspect_path, output_directional_slope_path)

    # Load directional slope GeoTIFF and save colorized PNGs
    with gdal.Open(output_directional_slope_path) as src:
        bands = src.ReadAsArray()  # Shape: (num_bands, height, width)
        bands = bands.transpose(1, 2, 0)  # Reshape to (height, width, num_bands)
        save_colorized_slope(bands.astype(np.float32), output_dir)

    # Colorize and save the aspect file as PNG
    colorized_aspect_png = os.path.join(output_dir, 'aspect.png')
    colorize_aspect(output_aspect_path, colorized_aspect_png)

def process_directories(root_dir):
    """Process all subdirectories containing original_topography.tif."""
    for dirpath, dirnames, filenames in os.walk(root_dir):
        if 'original_topography.tif' in filenames:
            input_dem_path = os.path.join(dirpath, 'original_topography.tif')
            print(f"Processing: {input_dem_path}")
            process_slope_and_aspect(input_dem_path)

# Specify the root directory
root_directory = 'Datasets/'  # Replace with your root directory path

# Process all subdirectories
process_directories(root_directory)

