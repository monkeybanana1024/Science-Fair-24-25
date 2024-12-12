import os
from osgeo import gdal
import numpy as np

def calculate_directional_slopes(input_path, output_path):
    # Open the DEM
    dem = gdal.Open(input_path)
    
    # Get DEM properties
    width = dem.RasterXSize
    height = dem.RasterYSize
    projection = dem.GetProjection()
    geotransform = dem.GetGeoTransform()
    
    # Create output dataset
    driver = gdal.GetDriverByName("GTiff")
    out_ds = driver.Create(output_path, width, height, 8, gdal.GDT_Float32)
    out_ds.SetProjection(projection)
    out_ds.SetGeoTransform(geotransform)
    
    # Calculate slope for 8 directions (N, NE, E, SE, S, SW, W, NW)
    directions = [0, 45, 90, 135, 180, 225, 270, 315]
    
    for i, direction in enumerate(directions):
        # Calculate slope in this direction
        options = gdal.DEMProcessingOptions(alg="ZevenbergenThorne", computeEdges=True)
        gdal.DEMProcessing("/vsimem/temp.tif", input_path, "slope", options=options)
        # Read the calculated slope
        temp_ds = gdal.Open("/vsimem/temp.tif")
        slope_array = temp_ds.GetRasterBand(1).ReadAsArray()
        
        # Write to the output band
        out_ds.GetRasterBand(i+1).WriteArray(slope_array)
        out_ds.GetRasterBand(i+1).SetDescription(f"Slope_{direction}")
    
    # Clean up
    out_ds = None
    gdal.Unlink("/vsimem/temp.tif")
    print(f"Omnidirectional slopes calculated for {input_path}")

def process_topography(root_dir):
    for subdir, dirs, files in os.walk(root_dir):
        for file in files:
            if file == 'topography.tif':
                input_path = os.path.join(subdir, file)
                output_path = os.path.join(subdir, 'slope.tif')
                
                try:
                    calculate_directional_slopes(input_path, output_path)
                except Exception as e:
                    print(f"Error processing {input_path}: {str(e)}")

# Specify the root directory where your numbered folders are located
root_directory = 'Datasets'  # Change this to your actual root directory

# Process all topography files
process_topography(root_directory)

print("Omnidirectional slope calculation completed for all subdirectories.")
