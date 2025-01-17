import ee
import os
import geemap
import requests
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, BarColumn, TextColumn, TimeRemainingColumn
from rich.panel import Panel

console = Console()

# Initialize Earth Engine
ee.Initialize(project='ee-sciencefair2425')

def read_coordinates(file_path):
    """Read coordinates from a text file and return as a list of tuples."""
    with open(file_path, 'r') as f:
        return [tuple(map(float, line.strip().split(','))) for line in f]

def create_aoi(lat, lon):
    """Create a 1x1 mile square (0.5-mile buffer around the point)."""
    center_point = ee.Geometry.Point([lon, lat])
    return center_point.buffer(804.672).bounds()

def process_directories(root_dir, coord_file, progress):
    """Process subdirectories and fetch Sentinel-2 imagery for each coordinate."""
    coordinates = read_coordinates(coord_file)
    subfolders = sorted([f for f in os.listdir(root_dir) if os.path.isdir(os.path.join(root_dir, f))], key=int)
    
    task = progress.add_task("[green]Processing coordinates", total=len(subfolders))

    for idx, subfolder in enumerate(subfolders):
        if idx >= len(coordinates):
            progress.print(f"[yellow]No coordinate available for subfolder: {subfolder}. Skipping.[/yellow]")
            progress.update(task, advance=1)
            continue
        
        lat, lon = coordinates[idx]
        aoi = create_aoi(lat, lon)

        found_image = False
        for year in range(2022, 2015, -1):
            sentinel2 = (ee.ImageCollection('COPERNICUS/S2_SR_HARMONIZED')
                    .filterBounds(aoi)
                    .filterDate(f'{year}-01-01', f'{year}-12-31')
                    .sort('system:time_start', False))
            
            if sentinel2.size().getInfo() > 0:
                found_image = True
                sentinel2_image = sentinel2.mosaic()
                break
        
        if not found_image:
            progress.print(f"[yellow]No Sentinel-2 imagery available for {subfolder} (Lat: {lat}, Lon: {lon}) within the last 6 years. Skipping.[/yellow]")
            progress.update(task, advance=1)
            continue
        
        # Select RGB bands for visualization (B4=Red, B3=Green, B2=Blue)
        rgb_image = sentinel2_image.select(['B4', 'B3', 'B2'])
        
        # Visualize with default min and max for RGB imagery
        vizParams = {'min': 0, 'max': 3000}
        rgb_image_vis = rgb_image.visualize(**vizParams)
        rgb_image_clipped = rgb_image_vis.clip(aoi)
        
        output_path_png = os.path.join(root_dir, subfolder, 'sat.png')
        
        url = rgb_image_clipped.getThumbURL({'dimensions': 1024, 'format': 'png', 'region': aoi})

        response = requests.get(url)
        if response.status_code == 200:
            with open(output_path_png, 'wb') as f:
                f.write(response.content)
            progress.print(f"[green]Exported Sentinel-2 RGB image to: {output_path_png}[/green]")
        else:
            progress.print(f"[bold red]Failed to download image for {subfolder}. Status code: {response.status_code}[/bold red]")
        
        progress.update(task, advance=1)

def main():
    root_directory = 'Datasets/'
    coordinates_file = 'Collection/coordinates.txt'

    console.print(Panel.fit("Sentinel-2 Imagery Processing for Landslide Remote Sensing", title="Project", border_style="bold blue"))
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

    console.print(Panel.fit("[bold green]All Sentinel-2 imagery processing completed successfully![/bold green]", title="Status", border_style="bold green"))

if __name__ == "__main__":
    main()
