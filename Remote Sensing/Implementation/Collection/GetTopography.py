import ee
import geemap
import os
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, BarColumn, TextColumn, TimeRemainingColumn
from rich.panel import Panel

console = Console()

def process_coordinate(lat, lon, ned, main_dir, sub_dir_num):
    center_lat, center_lon = float(lat), float(lon)
    center_point = ee.Geometry.Point([center_lon, center_lat])
    region = center_point.buffer(804.672).bounds()
    clipped_ned = ned.clip(region)
    output_dir = f'{main_dir}/{sub_dir_num}'
    os.makedirs(output_dir, exist_ok=True)
    output_path = f'{output_dir}/original_topography.tif'
    geemap.ee_export_image(
        clipped_ned, 
        filename=output_path,
        scale=30,  # Set the scale according to the GLO30 resolution (30 meters)
        region=region,
        file_per_band=False
    )

def main():
    ee.Initialize(project='ee-sciencefair2425')
    
    # Access the Copernicus GLO30 DEM collection
    # GLO30 is an ImageCollection, so we select the first image from the collection
    ned = ee.ImageCollection('COPERNICUS/DEM/GLO30').mean()
    
    with open('Collection/coordinates.txt', 'r') as file:
        coordinates = [line.strip().split(',') for line in file]
    
    main_dir = 'Datasets'
    os.makedirs(main_dir, exist_ok=True)

    console.print(Panel.fit("Landslide Remote Sensing AI Project", title="Project", border_style="bold blue"))
    console.print(f"[yellow]Total coordinates to process:[/yellow] [bold]{len(coordinates)}[/bold]")
    console.print(f"[yellow]Output directory:[/yellow] [bold]{os.path.abspath(main_dir)}[/bold]")

    progress = Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
        TimeRemainingColumn()
    )

    with progress:
        task = progress.add_task("[green]Processing coordinates", total=len(coordinates))
        
        for sub_dir_num, (lat, lon) in enumerate(coordinates, 1):
            try:
                process_coordinate(lat, lon, ned, main_dir, sub_dir_num)
                progress.update(task, advance=1)
            except Exception as e:
                console.print(f"[bold red]Error processing coordinates {lat}, {lon}: {e}[/bold red]")
                progress.update(task, advance=1)

    console.print(Panel.fit("[bold green]All downloads completed successfully![/bold green]", title="Status", border_style="bold green"))

if __name__ == "__main__":
    main()
