import ee
import os
import geemap
import numpy as np
from PIL import Image
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, BarColumn, TextColumn, TimeRemainingColumn
from rich.panel import Panel

console = Console()

# Initialize Earth Engine
ee.Initialize(project='ee-sciencefair2425')

def get_ndvi(image):
    """Calculate NDVI from NAIP imagery."""
    ndvi = image.normalizedDifference(['N', 'R']).rename('NDVI')  # N is Near Infrared, R is Red
    return image.addBands(ndvi)

def read_coordinates(file_path):
    """Read coordinates from a text file and return as a list of tuples."""
    with open(file_path, 'r') as f:
        lines = f.readlines()
    coordinates = []
    for line in lines:
        lat, lon = map(float, line.strip().split(','))
        coordinates.append((lat, lon))
    return coordinates

def create_aoi(lat, lon):
    """Create a 1x1 mile square (0.5-mile buffer around the point)."""
    center_point = ee.Geometry.Point([lon, lat])
    region = center_point.buffer(804.672).bounds()  # 804.672 meters = 0.5 miles
    return region

def apply_color_map(ndvi_array):
    """Apply a gradient color map to the NDVI array for better visualization."""
    # ... (rest of the function remains the same)

def process_directories(root_dir, coord_file, progress):
    """Process subdirectories and fetch NDVI for each coordinate."""
    coordinates = read_coordinates(coord_file)
    
    subfolders = next(os.walk(root_dir))[1]
    subfolders = sorted(subfolders, key=lambda x: int(x))
    
    task = progress.add_task("[green]Processing coordinates", total=len(subfolders))

    for idx, subfolder in enumerate(subfolders):
        if idx >= len(coordinates):
            progress.print(f"[yellow]No coordinate available for subfolder: {subfolder}. Skipping.[/yellow]")
            progress.update(task, advance=1)
            continue
        
        try:
            lat, lon = coordinates[idx]
            aoi = create_aoi(lat, lon)
            
            collection = (ee.ImageCollection('USDA/NAIP/DOQQ')
                          .filterBounds(aoi)
                          .filterDate('2018-01-01', '2022-12-31')
                          .map(get_ndvi))
            
            ndvi_image = collection.select('NDVI').median().clip(aoi)
            
            output_path_tif = os.path.join(root_dir, subfolder, 'ndvi.tif')
            
            geemap.ee_export_image(ndvi_image, filename=output_path_tif, scale=10, region=aoi)
            
            ndvi_array = np.array(Image.open(output_path_tif))
            colored_ndvi = apply_color_map(ndvi_array)
            output_path_png = os.path.join(root_dir, subfolder, 'ndvi.png')
            Image.fromarray(colored_ndvi).save(output_path_png)
            
            progress.update(task, advance=1)
        except Exception as e:
            progress.print(f"[bold red]Error processing subfolder {subfolder}: {e}[/bold red]")
            progress.update(task, advance=1)

def main():
    root_directory = 'Datasets/'
    coordinates_file = 'Collection/coordinates.txt'

    console.print(Panel.fit("NDVI Processing for Landslide Remote Sensing", title="Project", border_style="bold blue"))
    console.print(f"[yellow]Root directory:[/yellow] [bold]{os.path.abspath(root_directory)}[/bold]")
    console.print(f"[yellow]Coordinates file:[/yellow] [bold]{coordinates_file}[/bold]")

    progress = Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
        TimeRemainingColumn()
    )

    with progress:
        process_directories(root_directory, coordinates_file, progress)

    console.print(Panel.fit("[bold green]All NDVI processing completed successfully![/bold green]", title="Status", border_style="bold green"))

if __name__ == "__main__":
    main()
