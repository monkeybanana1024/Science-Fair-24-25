import os
import numpy as np
import rasterio
from PIL import Image
import matplotlib.pyplot as plt

def normalize_tiff(input_path, output_path):
    """Normalize the TIFF data and save it as a new TIFF file."""
    with rasterio.open(input_path) as src:
        # Read the data
        data = src.read(1).astype(float)
        
        # Normalize the data
        min_val = np.min(data)
        max_val = np.max(data)
        normalized_data = ((data - min_val) / (max_val - min_val)) * 255
        
        # Create a new raster file with normalized data
        profile = src.profile
        profile.update(dtype=rasterio.float32)
        
        with rasterio.open(output_path, 'w', **profile) as dst:
            dst.write(normalized_data.astype(rasterio.float32), 1)
        
def create_color_gradient_png(input_path, png_output_path):
    """Create a color gradient PNG from the normalized TIFF data."""
    with rasterio.open(input_path) as src:
        # Read the normalized data
        normalized_data = src.read(1).astype(float)
        
        # Normalize for color mapping (0-1 range)
        min_val = np.min(normalized_data)
        max_val = np.max(normalized_data)
        normalized_for_color = (normalized_data - min_val) / (max_val - min_val)

        # Create a color gradient (green to red)
        colormap = plt.get_cmap('RdYlGn')  # Green to Red colormap

        # Apply colormap and convert to RGB
        colored_image = colormap(normalized_for_color)[:, :, :3]  # Get RGB channels
        colored_image = (colored_image * 255).astype(np.uint8)  # Convert to uint8

        # Save as PNG
        img = Image.fromarray(colored_image)
        img.save(png_output_path)

def process_directories(root_dir):
    for dirpath, dirnames, filenames in os.walk(root_dir):
        for filename in filenames:
            if filename == 'original_topography.tif':
                input_path = os.path.join(dirpath, filename)
                output_path = os.path.join(dirpath, 'normalized_topography.tif')
                png_output_path = os.path.join(dirpath, 'normalized_topography.png')
                
                # Normalize TIFF and create PNG with color gradient
                normalize_tiff(input_path, output_path)
                create_color_gradient_png(output_path, png_output_path)
                
                print(f"Processed: {input_path}")
                print(f"Created Normalized TIFF: {output_path}")
                print(f"Created Color Gradient PNG: {png_output_path}")

# Specify the root directory where your subdirectories are located
root_directory = 'Datasets/'

# Process all subdirectories
process_directories(root_directory)
