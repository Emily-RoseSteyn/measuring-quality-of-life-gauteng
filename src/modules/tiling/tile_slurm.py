import os
import time
from pathlib import Path

from modules.tiling.get_tile_list import get_tile_list
from modules.tiling.tile_image import tile_image
from mpi4py import MPI
from utils.logger import get_logger

logger = get_logger()


def tile_slurm(file_list: list, output_directory: str) -> None:
    logger.info("Tiling with slurm")

    start_time = time.time()
    rank = MPI.COMM_WORLD.Get_rank()
    size = MPI.COMM_WORLD.Get_size()
    name = MPI.Get_processor_name()

    # Might want to setup some stuff here
    if rank == 0:
        logger.debug("I'm rank 0")

    # Make sure rank 0 has done its stuff before moving on
    MPI.COMM_WORLD.Barrier()

    # For each file
    for index, item in enumerate(file_list):
        # If there are still files to process and processor is ready
        if index % size != rank:
            continue

        logger.info(
            "Item %s being done by processor %d (%s) of %d", item, rank, name, size
        )

        # Do stuff here!
        tile_image(item, output_directory, index)

    # End do stuff

    # Finished
    logger.info(
        "Node %s time spent in minutes: %s",
        ((rank - 1) % size),
        int(time.time() - start_time) / 60,
    )


def main() -> None:
    images = get_tile_list()
    results_dir = os.path.abspath(Path("./outputs/tiles"))
    tile_slurm(images, results_dir)


if __name__ == "__main__":
    main()
