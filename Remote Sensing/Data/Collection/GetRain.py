import ee
import geemap
import os
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, BarColumn, TextColumn, TimeRemainingColumn
from rich.panel import Panel

console = Console()

def process_coordinate(lat, lon, chirps, main_dir, sub_dir_num):
    center_lat, center_lon = float(lat), float(lon)
    center_point = ee.Geometry.Point([center_lon, center_lat])
    region = center_point.buffer(804.672).bounds()

    mean_precipitation = chirps.filterBounds(region).mean().reduceRegion(
        reducer=ee.Reducer.mean(),
        geometry=region,
        scale=5000,
        maxPixels=1e13
    )

    avg_precip_value = mean_precipitation.get('precipitation')
    avg_precip_value_result = avg_precip_value.getInfo()

    rain_file_path = f'{main_dir}/{sub_dir_num}/rain.txt'
    os.makedirs(os.path.dirname(rain_file_path), exist_ok=True)

    with open(rain_file_path, 'w') as rain_file:
        rain_file.write(str(avg_precip_value_result))

def main():
    ee.Initialize(project='ee-sciencefair2425')
    chirps = ee.ImageCollection('UCSB-CHG/CHIRPS/DAILY')
    
    with open('Collection/coordinates.txt', 'r') as file:
        coordinates = [line.strip().split(',') for line in file]
    
    main_dir = 'Datasets'

    console.print(Panel.fit("CHIRPS Precipitation Processing", title="Project", border_style="bold blue"))
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
                process_coordinate(lat, lon, chirps, main_dir, sub_dir_num)
                progress.update(task, advance=1)
            except Exception as e:
                console.print(f"[bold red]Error processing coordinates {lat}, {lon}: {e}[/bold red]")
                progress.update(task, advance=1)

    console.print(Panel.fit("[bold green]All precipitation data processing completed successfully![/bold green]", title="Status", border_style="bold green"))

if __name__ == "__main__":
    main()
