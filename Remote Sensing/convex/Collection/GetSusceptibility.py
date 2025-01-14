import os
import rasterio
from rich.progress import Progress, SpinnerColumn, BarColumn, TextColumn, TimeRemainingColumn
from rich.console import Console

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
    console = Console()
    
    # Path to your TIFF file
    tiff_file = '../Data/Collection/n10_conus.tif'  # Update this path
    ensure_file_exists(tiff_file)
    
    console.print(os.listdir())
    
    # Create Datasets directory if it doesn't exist
    datasets_dir = 'Datasets/'
    if not os.path.exists(datasets_dir):
        os.makedirs(datasets_dir)
    
    # Read coordinates from the file
    coordinates_file = 'Collection/coordinates.txt'
    ensure_file_exists(coordinates_file)
    
    with open(coordinates_file, 'r') as coord_file:
        coordinates = coord_file.readlines()

    # Create folders based on the number of coordinates
    for i in range(1, len(coordinates) + 1):
        os.makedirs(os.path.join(datasets_dir, str(i)), exist_ok=True)

    # Process each coordinate with Rich progress bar
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
        TimeRemainingColumn()
    ) as progress:
        task = progress.add_task("[green]Processing coordinates", total=len(coordinates))
        
        for index, line in enumerate(coordinates, start=1):
            try:
                lat, long = map(float, line.strip().split(','))
                value = analyze_tiff(lat, long, tiff_file)

                output_file_path = os.path.join(datasets_dir, str(index), 'susc.txt')

                # Ensure the output file exists
                ensure_file_exists(output_file_path)

                # Write the result to susc.txt in the appropriate folder
                with open(output_file_path, 'w') as output_file:
                    output_file.write(str(value))

                progress.update(task, advance=1)

            except ValueError as e:
                console.print(f"\nError processing line {index}: {line.strip()} - {e}", style="bold red")
            except IndexError as e:
                console.print(f"\nCoordinate out of bounds for line {index}: {line.strip()} - {e}", style="bold red")
            except Exception as e:
                console.print(f"\nUnexpected error processing line {index}: {line.strip()} - {e}", style="bold red")

    console.print("\nAll coordinates processed.", style="bold green")

if __name__ == "__main__":
    main()
