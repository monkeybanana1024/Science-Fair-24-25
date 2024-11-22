import sys
import os

# Add the OpenTopoData directory to the Python path
# Replace '/path/to/opentopodata' with the actual path where you cloned the OpenTopoData repository
sys.path.append(os.path.abspath("opentopodata/"))

from opentopodata import backend

def get_elevation(lat, lon, dataset_path):
    # Load the dataset
    dataset = backend.load_dataset(dataset_path)
    
    # Query elevation
    elevation = backend.get_elevation(dataset, lat, lon)
    
    return elevation

# Example usage
if __name__ == "__main__":
    # Replace this with the path to your elevation data
    dataset_path = "/path/to/your/elevation/data"
    
    # Example coordinates
    latitude = 56.0
    longitude = 123.0

    elevation = get_elevation(latitude, longitude, dataset_path)
    
    if elevation is not None:
        print(f"Elevation at {latitude}, {longitude}: {elevation} meters")
    else:
        print("Failed to retrieve elevation data")