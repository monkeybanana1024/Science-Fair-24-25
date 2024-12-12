import ee
import geemap

# Initialize Earth Engine
ee.Initialize(project='ee-sciencefair2425')

# Load the NED 10m dataset
ned = ee.Image('USGS/3DEP/10m')

# Read coordinates from file
with open('coordinates.txt', 'r') as file:
    coordinates = [line.strip().split(',') for line in file]

# Main directory
main_dir = 'Datasets'

# Starting subdirectory number
sub_dir_num = 1

# Process each coordinate pair
for lat, lon in coordinates:
    # Convert coordinates to float, swapping the order
    center_lat, center_lon = float(lat), float(lon)

    # Create a point geometry for the center
    center_point = ee.Geometry.Point([center_lon, center_lat])

    # Create a 1x1 mile square region around the center point
    region = center_point.buffer(804.672).bounds()  # 804.672 meters is half a mile

    # Clip the NED dataset to the region
    clipped_ned = ned.clip(region)

    # Define the output path
    output_path = f'{main_dir}/{sub_dir_num}/topography.tif'

    # Use geemap to download the image
    geemap.ee_export_image(
        clipped_ned, 
        filename=output_path,
        scale=10,
        region=region,
        file_per_band=False
    )

    print(f"GeoTIFF for coordinates {center_lat}, {center_lon} has been downloaded to {output_path}")

    # Increment the subdirectory number
    sub_dir_num += 1

print("All downloads completed.") 