import os
import subprocess
from pathlib import Path
from typing import Generator

from modules.tiling.tile_image import tile_image
from utils.env_variables import SLURM_ENABLED
from utils.logger import get_logger


def absolute_file_paths(directory: str) -> Generator[str, None, None]:
    for dirpath, _, filenames in os.walk(directory):
        for f in filenames:
            if f.endswith(".tiff"):
                yield os.path.abspath(os.path.join(dirpath, f))


def tile_without_slurm(file_list: list, output_directory: str) -> None:
    for file in file_list:
        tile_image(file, output_directory)


def main() -> None:
    logger = get_logger()
    logger.info("In tile images")

    # Output directory
    results_dir = os.path.abspath(Path("./outputs/tiles"))
    if not os.path.isdir(results_dir):
        os.makedirs(results_dir)

    # Get images directory
    image_dir = "data/basemap-quads"
    images = list(absolute_file_paths(image_dir))

    # TODO: Slurm flow changes
    #  - Enable slurm on cluster
    #  - Pass commands into script (images and results dir)
    #  - Check if W works?
    if SLURM_ENABLED:
        # If so, dispatch slurm script with wait
        logger.info("Running with SLURM")
        sbatch_script = Path("./src/modules/tiling/tile.sbatch")
        subprocess.call(f"cat {sbatch_script}")  # noqa: S603
    else:
        # If not, tile individually sequentially
        logger.info("SLURM is not available")
        tile_without_slurm(images, results_dir)

    # TODO: How to save corresponding qol data?
    with open(f"{results_dir}/gauteng-qol-tile-map.csv", "w"):
        logger.info("Doing nothing for now")


if __name__ == "__main__":
    main()
