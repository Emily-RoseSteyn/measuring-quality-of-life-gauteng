import os
import subprocess
from pathlib import Path

from modules.tiling.tile_image import tile_image
from utils.env_variables import SLURM_ENABLED
from utils.file_utils import dir_nested_file_list
from utils.logger import get_logger


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
    images = list(dir_nested_file_list(image_dir, "tiff"))

    # TODO: Slurm flow changes
    #  - Enable slurm on cluster
    #  - Pass commands into script (images and results dir)
    #  - Check if W works?
    if SLURM_ENABLED:
        # If so, dispatch slurm script with wait
        logger.info("Running with SLURM")
        sbatch_script = Path("./src/modules/tiling/tile.sbatch")
        cmd = f"sbatch {sbatch_script} -W"
        subprocess.call(cmd.split())  # noqa: S603
    else:
        # If not, tile individually sequentially
        logger.info("SLURM is not available")
        tile_without_slurm(images, results_dir)


if __name__ == "__main__":
    main()
