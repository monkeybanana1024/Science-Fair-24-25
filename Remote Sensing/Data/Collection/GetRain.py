import ee
import geemap

# Initialize Earth Engine
ee.Initialize(project='ee-sciencefair2425')

# Load the CHIRPS dataset
chirps = ee.ImageCollection('UCSB-CHG/CHIRPS/DAILY')

# Read coordinates from file
with open('Collection/coordinates.txt', 'r') as file:
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

    # Calculate average precipitation from CHIRPS for the region
    mean_precipitation = chirps.filterBounds(region).mean().reduceRegion(
        reducer=ee.Reducer.mean(),
        geometry=region,
        scale=5000,  # Adjust scale as necessary for your area of interest
        maxPixels=1e13
    )

    # Get the average precipitation value from the result
    avg_precip_value = mean_precipitation.get('precipitation')

    # Wait for the value to be computed and retrieve it
    avg_precip_value_result = avg_precip_value.getInfo()  # This retrieves the value from Earth Engine

    # Define the output path for rain.txt
    rain_file_path = f'{main_dir}/{sub_dir_num}/rain.txt'

    # Write the average precipitation value to rain.txt
    with open(rain_file_path, 'w') as rain_file:
        rain_file.write(str(avg_precip_value_result))

    print(f"Average precipitation has been written to {rain_file_path}")

    # Increment the subdirectory number
    sub_dir_num += 1

print("All downloads completed.")
