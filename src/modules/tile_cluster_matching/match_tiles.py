import os
from pathlib import Path

import rasterio as rio
from utils.file_utils import dir_nested_file_list
from utils.logger import get_logger


def main() -> None:
    logger = get_logger()
    logger.info("Matching tiles to clusters")

    # Output & tiles directory
    results_dir = os.path.abspath(Path("./outputs/tiles"))
    if not os.path.isdir(results_dir):
        os.makedirs(results_dir)

    # Get tiles
    tiles = list(dir_nested_file_list(results_dir, "tiff"))

    # TODO: Parallelize me
    for tile in tiles:
        with rio.open(tile) as img_object:
            logger.debug(img_object.crs)


if __name__ == "__main__":
    main()
