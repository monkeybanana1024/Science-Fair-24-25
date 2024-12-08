import ee
import requests

# Initialize the Earth Engine module
ee.Initialize(project='ee-sciencefair2425')

# Function to read coordinates from a file
def read_coordinates(file_path):
    with open(file_path, 'r') as file:
        lines = file.readlines()
        # Assuming the file contains one coordinate pair per line in "lat,lon" format
        coordinates = [tuple(map(float, line.strip().split(','))) for line in lines]
    return coordinates

# Read coordinates from the specified file
coordinates = read_coordinates('coordinates.txt')

# Loop through each coordinate pair and process elevation data
for lat, lon in coordinates:
    # Create a point from the coordinates
    point = ee.Geometry.Point(lon, lat)

    # Define a 1 mile x 1 mile square around the point
    buffer_distance = 804.67  # Half of 1 mile in meters
    roi = point.buffer(buffer_distance).bounds()

    # Load the Copernicus Digital Elevation Model (DEM) as an ImageCollection
    copernicus_dem = ee.ImageCollection("COPERNICUS/DEM/GLO30")

    # Check if there are any images in the collection
    if copernicus_dem.size().getInfo() == 0:
        print(f'No images found in Copernicus DEM for coordinates: {lat}, {lon}.')
        continue

    # Use mosaic to combine images and clip to ROI
    elevation_data = copernicus_dem.mosaic().clip(roi)

    # Generate a download URL for the image
    download_url = elevation_data.getDownloadURL({
        'scale': 30,  # Scale in meters
        'region': roi.getInfo()['coordinates'],
        'format': 'GeoTIFF'
    })

    # Download the GeoTIFF file
    response = requests.get(download_url)

    if response.status_code == 200:
        # Save the file locally with a unique name based on coordinates
        filename = f'elevation_data_{lat}_{lon}.tif'
        with open(filename, 'wb') as f:
            f.write(response.content)
        print(f'Downloaded: {filename}')
    else:
        print(f'Failed to download data for coordinates: {lat}, {lon}. Status code: {response.status_code}')