import numpy as np
import matplotlib.pyplot as plt
from rasterio.enums import Resampling
import rasterio

# Path to your TIFF file
tiff_file_path = 'USGS-susc/n10_conus.tif'

# Open the TIFF file
with rasterio.open(tiff_file_path) as src:
    # Read the data into a memory-mapped array
    data = src.read(1, out_shape=(src.height, src.width), 
                    resampling=Resampling.nearest)

    # Get the metadata
    transform = src.transform
    crs = src.crs

# Create a figure and axis
fig, ax = plt.subplots(figsize=(10, 10))

# Display the image
im = ax.imshow(data, cmap='terrain')

# Add a colorbar
plt.colorbar(im, ax=ax, label='Elevation')

# Set title and labels
ax.set_title('Elevation Map')
ax.set_xlabel('X')
ax.set_ylabel('Y')

# Show the plot
plt.show()