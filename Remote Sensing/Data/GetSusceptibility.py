import os
import rasterio

def analyze_tiff(lat, long, tiff_file):
    """
    Analyze the TIFF file for the given latitude and longitude.
    
    Args:
        lat (float): Latitude coordinate.
        long (float): Longitude coordinate.
        tiff_file (str): Path to the TIFF file.
    
    Returns:
        float: Value extracted from the TIFF at the specified coordinates.
    """
    with rasterio.open(tiff_file) as src:
        # Convert lat/long to row/column indices
        row, col = src.index(long, lat)
        
        # Read the value from the TIFF at that location
        value = src.read(1)[row, col]  # Read the first band
    return value

def ensure_file_exists(file_path):
    directory = os.path.dirname(file_path)
    if directory:
        os.makedirs(directory, exist_ok=True)
    if not os.path.exists(file_path):
        open(file_path, 'a').close()
        
def main():
    # Path to your TIFF file
    tiff_file = 'n10_conus.tif'  # Update this path
    ensure_file_exists(tiff_file)
    
    print(os.listdir())
    
    # Create Test directory if it doesn't exist
    test_dir = 'Test'
    if not os.path.exists(test_dir):
        os.makedirs(test_dir)
    
    os.chdir(test_dir)
    
    # Create directories from 1 to 100
    for i in range(1, 101):
        os.makedirs(str(i), exist_ok=True)

    # Read coordinates from the file
    os.chdir("..")
    coordinates_file = 'coordinates.txt'
    ensure_file_exists(coordinates_file)
    
    with open(coordinates_file, 'r') as coord_file:
        coordinates = coord_file.readlines()

    # Process each coordinate
    for index, line in enumerate(coordinates):
        try:
            lat, long = map(float, line.strip().split(','))
            value = analyze_tiff(lat, long, tiff_file)

            # Determine the folder based on index (1-100)
            folder_number = (index % 100) + 1
            output_file_path = os.path.join(test_dir, str(folder_number), 'susc.txt')

            # Ensure the output file exists
            ensure_file_exists(output_file_path)

            # Write the result to susc.txt in the appropriate folder
            with open(output_file_path, 'a') as output_file:
                output_file.write(f"{lat}, {long}: {value}\n")

        except ValueError as e:
            print(f"Error processing line {index + 1}: {line.strip()} - {e}")
        except IndexError as e:
            print(f"Coordinate out of bounds for line {index + 1}: {line.strip()} - {e}")
        except Exception as e:
            print(f"Unexpected error processing line {index + 1}: {line.strip()} - {e}")

if __name__ == "__main__":
    main()