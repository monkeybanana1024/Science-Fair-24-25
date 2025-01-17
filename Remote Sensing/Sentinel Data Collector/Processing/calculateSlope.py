import os
import numpy as np
from osgeo import gdal
import matplotlib.pyplot as plt
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, BarColumn, TextColumn, TimeRemainingColumn
from rich.panel import Panel

console = Console()

def calculate_slope_and_aspect(input_dem_path, output_slope_path, output_aspect_path):
    """Calculate slope and aspect from a DEM using GDAL."""
    cmd_slope = f"gdaldem slope -compute_edges {input_dem_path} {output_slope_path}"
    cmd_aspect = f"gdaldem aspect -compute_edges {input_dem_path} {output_aspect_path}"
    os.system(cmd_slope)
    os.system(cmd_aspect)

def create_directional_slope(slope_file, aspect_file, output_file):
    """Create an 8-band directional slope raster."""
    slope_ds = gdal.Open(slope_file)
    aspect_ds = gdal.Open(aspect_file)
    
    slope = slope_ds.GetRasterBand(1).ReadAsArray()
    aspect = aspect_ds.GetRasterBand(1).ReadAsArray()
    
    aspect_rad = np.deg2rad(aspect)
    
    bands = []
    for i in range(8):
        angle = i * 45
        weight = np.cos(aspect_rad - np.deg2rad(angle))
        directional_slope = slope * weight
        bands.append(directional_slope)
    
    bands_normalized = [(band - np.min(band)) / (np.max(band) - np.min(band)) * 255 for band in bands]
    
    driver = gdal.GetDriverByName('GTiff')
    out_ds = driver.Create(output_file, slope_ds.RasterXSize, slope_ds.RasterYSize, 8, gdal.GDT_Byte)
    
    out_ds.SetGeoTransform(slope_ds.GetGeoTransform())
    out_ds.SetProjection(slope_ds.GetProjection())
    
    for i, band in enumerate(bands_normalized):
        out_band = out_ds.GetRasterBand(i + 1)
        out_band.WriteArray(band.astype(np.uint8))
    
    out_ds = None
    slope_ds = None
    aspect_ds = None

def colorize_slope(slope_band):
    """Colorize a single slope band based on its values."""
    cmap = plt.get_cmap('RdYlGn')
    normalized_band = (slope_band - np.min(slope_band)) / (np.max(slope_band) - np.min(slope_band))
    colored_image = cmap(normalized_band)[:, :, :3]
    return (colored_image * 255).astype(np.uint8)

def save_colorized_slope(slope_data, output_dir):
    """Save colorized slope images for each direction."""
    directions = ['N', 'NE', 'E', 'SE', 'S', 'SW', 'W', 'NW']
    
    for i in range(slope_data.shape[2]):
        colored_image = colorize_slope(slope_data[:, :, i])
        output_filename = os.path.join(output_dir, f'{directions[i]}.png')
        plt.imsave(output_filename, colored_image)

def colorize_aspect(aspect_file, output_png_path):
    """Colorize the aspect raster using a circular colormap."""
    aspect_ds = gdal.Open(aspect_file)
    aspect = aspect_ds.GetRasterBand(1).ReadAsArray()

    normalized_aspect = np.where(aspect >= 0, aspect / 360.0, np.nan)
    cmap = plt.cm.hsv

    colored_aspect = cmap(normalized_aspect)[:, :, :3]

    flat_color = [0.5, 0.5, 0.5]
    nan_mask = np.isnan(normalized_aspect)
    colored_aspect[nan_mask] = flat_color

    colored_aspect_uint8 = (colored_aspect * 255).astype(np.uint8)
    plt.imsave(output_png_path, colored_aspect_uint8)

def process_slope_and_aspect(input_dem_path):
    """Process DEM to calculate slope and aspect, and colorize both."""
    output_dir = os.path.join(os.path.dirname(input_dem_path), 'slope')
    os.makedirs(output_dir, exist_ok=True)

    output_slope_path = os.path.join(output_dir, 'slope.tif')
    output_aspect_path = os.path.join(output_dir, 'aspect.tif')
    
    calculate_slope_and_aspect(input_dem_path, output_slope_path, output_aspect_path)

    output_directional_slope_path = os.path.join(output_dir, 'slope.tif')
    create_directional_slope(output_slope_path, output_aspect_path, output_directional_slope_path)

    with gdal.Open(output_directional_slope_path) as src:
        bands = src.ReadAsArray()
        bands = bands.transpose(1, 2, 0)
        save_colorized_slope(bands.astype(np.float32), output_dir)

    colorized_aspect_png = os.path.join(output_dir, 'aspect.png')
    colorize_aspect(output_aspect_path, colorized_aspect_png)

def process_directories(root_dir, progress):
    """Process all subdirectories containing original_topography.tif."""
    total_files = sum([1 for r, d, f in os.walk(root_dir) if 'original_topography.tif' in f])
    task = progress.add_task("[green]Processing DEMs", total=total_files)

    for dirpath, dirnames, filenames in os.walk(root_dir):
        if 'original_topography.tif' in filenames:
            input_dem_path = os.path.join(dirpath, 'original_topography.tif')
            try:
                process_slope_and_aspect(input_dem_path)
                progress.update(task, advance=1)
            except Exception as e:
                progress.print(f"[bold red]Error processing {input_dem_path}: {e}[/bold red]")
                progress.update(task, advance=1)

def main():
    root_directory = 'Datasets/'  # Replace with your root directory path

    console.print(Panel.fit("DEM Slope and Aspect Processing", title="Project", border_style="bold blue"))
    console.print(f"[yellow]Root directory:[/yellow] [bold]{os.path.abspath(root_directory)}[/bold]")

    progress = Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
        TimeRemainingColumn()
    )

    with progress:
        process_directories(root_directory, progress)

    console.print(Panel.fit("[bold green]All DEMs processed successfully![/bold green]", title="Status", border_style="bold green"))

if __name__ == "__main__":
    main()
