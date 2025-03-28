import os
from itertools import product
from pathlib import Path
from time import strptime
from typing import Any

import geopandas as gpd
import rasterio as rio
from dvc.api import params_show
from rasterio import windows
from shapely import box

from utils.logger import get_logger

logger = get_logger()

# Get DVC params
params = params_show()


# TODO: Understand this more
# Based on this SO:
# https://gis.stackexchange.com/questions/285499/how-to-split-multiband-image-into-image-tiles-using-rasterio
def get_tiles(image: Any, crop_size: int) -> Any:
    # Assuming image is a square and tile is a square too
    height, width = image.meta["height"], image.meta["width"]
    # Using range, get columns and rows spaced by crop_size
    cols = range(0, width, crop_size)
    rows = range(0, height, crop_size)
    # Get grid
    grid = product(cols, rows)
    big_window = windows.Window(col_off=0, row_off=0, width=width, height=height)

    # For each col/row step in the grid
    for col_off, row_off in grid:
        # Get the window
        window = windows.Window(
            col_off=col_off, row_off=row_off, width=crop_size, height=crop_size
        ).intersection(big_window)
        # Apply the window to the original datasource to get the correct transform
        transform = windows.transform(window, image.transform)

        yield window, transform


def tile_image(file_path: str, output_dir: str, thread: int = 0) -> None:
    logger.info("Tiling %s", file_path)
    path = Path(file_path)
    # Extract date of tiff
    date = path.parent.name
    basename = Path(os.path.basename(path))
    file_name = basename.stem
    suffix = basename.suffix

    # Crop size from params
    crop_size = params["preprocessing"]["crop_size"]

    tile_transforms = []
    with rio.open(file_path) as img_object:
        metadata = img_object.meta.copy()
        for window, transform in get_tiles(img_object, crop_size):
            # Setting metadata
            metadata["transform"] = transform
            metadata["width"], metadata["height"] = window.width, window.height

            # Getting tile "number"
            index_x = str(int(window.col_off / crop_size)).zfill(2)
            index_y = str(int(window.row_off / crop_size)).zfill(2)
            tile_number = f"{index_x}_{index_y}"

            # Setting file name
            tile_file_name = f"{date}-{file_name}_{tile_number}{suffix}"
            output_file_path = Path.joinpath(Path(output_dir), tile_file_name)

            # Writing file
            img_read = img_object.read(window=window)
            with rio.open(output_file_path, "w", **metadata) as dest:
                # Read the original image object windowed by the current tile window
                dest.write(img_read)

                # Appending bounds of tile to dataframe to store in geojson
                bounds = dest.bounds
                geom = box(*bounds)
                # geom = box(*windows.bounds(window, transform))
                tile_transforms.append({"geometry": geom, "tile": tile_file_name})

        # Storing in geojson to merge in parent
        tile_transforms_df = gpd.GeoDataFrame(tile_transforms, crs=img_object.crs)
        year = strptime(date, "%Y-%m").tm_year
        tile_transforms_df["year"] = f"{year}"
        tile_transforms_df.to_file(
            os.path.join(output_dir, f"tile-transforms_{thread}.geojson"),
            driver="GeoJSON",
            mode="w",
        )
