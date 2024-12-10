import os
from osgeo import gdal

def process_topography(root_dir):
    for subdir, dirs, files in os.walk(root_dir):
        for file in files:
            if file == 'topography.tif':
                input_path = os.path.join(subdir, file)
                output_path = os.path.join(subdir, 'slope.tif')
                
                try:
                    # Calculate slope using GDAL
                    gdal.DEMProcessing(output_path, input_path, 'slope')
                    print(f"Slope calculated for {input_path}")
                except Exception as e:
                    print(f"Error processing {input_path}: {str(e)}")

# Specify the root directory where your numbered folders are located
root_directory = 'Test'  # Change this to your actual root directory

# Process all topography files
process_topography(root_directory)

print("Slope calculation completed for all subdirectories.")