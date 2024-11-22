import sys
import os
from opentopodata.opentopodata import backend  # Adjust based on your structure

# Add the parent directory of opentopodata to the Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Define a simple Dataset class for demonstration purposes
class Dataset:
    def __init__(self, name, vrt_path):
        self.name = name
        self.vrt_path = vrt_path

    def location_paths(self, lats, lons):
        # This method should return the paths for each latitude/longitude pair.
        # For simplicity, returning the same VRT path for all points.
        return [self.vrt_path] * len(lats)

    @property
    def wgs84_bounds(self):
        # You should define the bounds of your dataset here.
        return {'bottom': 55.0, 'top': 57.0, 'left': 122.0, 'right': 124.0}

def get_elevation(lats, lons, datasets, interpolation="nearest", nodata_value=None):
    elevations, dataset_names = backend.get_elevation(lats, lons, datasets, interpolation, nodata_value)
    return elevations

if __name__ == "__main__":
    # Example coordinates
    lats = [56.0]
    lons = [123.0]

    # Define your VRT file path
    vrt_file_path = "opentopodata/data/ned10m/ned10m.vrt"  # Update this path accordingly

    # Create a dataset instance
    datasets = [Dataset("NED10M", vrt_file_path)]

    try:
        elevations = get_elevation(lats, lons, datasets)
        
        if elevations:
            for lat, lon, elevation in zip(lats, lons, elevations):
                print(f"Elevation at {lat}, {lon}: {elevation} meters")
        else:
            print("Failed to retrieve elevation data")
    except Exception as e:
        print(f"An error occurred: {e}")