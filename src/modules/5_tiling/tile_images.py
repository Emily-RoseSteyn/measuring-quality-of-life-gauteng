import os
import subprocess
from pathlib import Path

from modules.tiling.get_tile_list import get_tile_list
from modules.tiling.merge_geojson import merge_geojson
from modules.tiling.tile_image import tile_image
from tqdm import tqdm

from utils.env_variables import SLURM_ENABLED
from utils.logger import get_logger


def tile_without_slurm(file_list: list, output_directory: str) -> None:
    for index, file in enumerate(tqdm(file_list)):
        tile_image(file, output_directory, thread=index)


def main() -> None:
    logger = get_logger()
    logger.info("In tile images")

    # Output directory
    results_dir = os.path.abspath(Path("./outputs/tiles"))
    if not os.path.isdir(results_dir):
        os.makedirs(results_dir)

    # TODO: Slurm flow changes
    #  - Check if W works?
    if SLURM_ENABLED:
        # If so, dispatch slurm script with wait
        logger.info("Running with SLURM")
        sbatch_script = Path("./src/modules/5_tiling/tile.sbatch")
        cmd = f"sbatch {sbatch_script} -W"
        subprocess.call(cmd.split())  # noqa: S603
    else:
        # If not, tile individually sequentially
        logger.info("SLURM is not available")
        images = get_tile_list()
        tile_without_slurm(images, results_dir)

        # When done, merge geojson data
        merge_geojson(results_dir)


if __name__ == "__main__":
    main()
