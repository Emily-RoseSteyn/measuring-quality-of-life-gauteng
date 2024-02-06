import os
from pathlib import Path
from typing import Any

import numpy as np
import rasterio as rio
from earthpy.spatial import crs_check
from utils.logger import get_logger

logger = get_logger()


def read_image(file_name: str) -> Any:
    """
    Read image from file_name
    Args: file_name: image file name
    Returns: image array
    """
    try:
        if not Path(file_name).is_file():
            logger.error("Cannot open file! %s", file_name)
            return np.array([])
        with rio.open(file_name) as img_object:
            img = img_object.read()
            return img
    except Exception:
        logger.exception("Error in read_image: %s")
        return np.array([])


def save_image(img_arr: np.ndarray, output_file_path: Path, crs: str) -> Path:
    """
    Save image to file_name
    Args: img_arr: image array
    Output: output_file_path: image file name
    """
    with rio.open(
        output_file_path,
        "w",
        driver="GTiff",
        count=img_arr.shape[0],
        height=img_arr.shape[1],
        width=img_arr.shape[2],
        dtype=img_arr.dtype,
        crs=crs,
    ) as dest:
        dest.write(img_arr)
    return output_file_path


def tile_image(file_path: str, output_dir: str, crop_size: int = 256) -> None:
    logger.info("Tiling %s", file_path)

    # Get CRS
    crs = crs_check(file_path)

    # Read Image
    image = read_image(file_path)

    # Ensuring image is in "contiguous C ordered arrays with channels at the lowest dimension"
    # This is important for the reshaping below
    image = np.moveaxis(image, 0, 2)
    img_height, img_width, channels = image.shape

    # Assuming image is a square
    tile_height, tile_width = (crop_size, crop_size)

    # Reshape image
    # TODO: Understand this better
    tiled_array = image.reshape(
        img_height // tile_height,
        tile_height,
        img_width // tile_width,
        tile_width,
        channels,
    )

    # Making sure tiles are ordered by row tiles, column tiles, channels, row, height
    # ie of shape 16 x 16 x 4 x 256 x256
    tiled_array = tiled_array.swapaxes(1, 2)
    tiled_array = np.moveaxis(tiled_array, 4, 2)

    for i, row in enumerate(tiled_array):
        for j, tile in enumerate(row):
            basename = Path(os.path.basename(file_path))
            file_name = f"{basename.stem}_{i}{j}{basename.suffix}"
            output_file_path = Path.joinpath(Path(output_dir), file_name)
            save_image(tile, output_file_path, crs)

    logger.info("Output dir %s", output_dir)
