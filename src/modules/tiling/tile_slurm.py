import subprocess
import time

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
    for item in file_list:
        # If there are still files to process and processor is ready
        if (item - 1) % size != rank:
            continue

        logger.info(
            "Item %s being done by processor %d (%s) of %d", item, rank, name, size
        )

        # Do stuff here!
        tile_image(item, output_directory)

    # End do stuff

    # Finished
    logger.info(
        "Node %s time spent in minutes: %s",
        ((rank - 1) % size),
        int(time.time() - start_time) / 60,
    )


if __name__ == "__main__":
    # Some sort of list of things we want to do stuff with
    allitems = list(range(1, 128))

    # mpi4py
    start_time = time.time()
    rank = MPI.COMM_WORLD.Get_rank()
    size = MPI.COMM_WORLD.Get_size()
    name = MPI.Get_processor_name()

    # Might want to setup some stuff here
    if rank == 0:
        logger.info("I'm rank 0")

    # Make sure rank 0 has done its stuff before moving on
    MPI.COMM_WORLD.Barrier()

    # RUN SIMULATION
    for item in allitems:
        # here lies the magic
        if (item - 1) % size != rank:
            continue

        logger.info(
            "Item %s being done by processor %d (%s) of %d", item, rank, name, size
        )

        # Do stuff here!
        # in this template an artificial cpu stress is run for 20 seconds
        cmd = "stress --cpu 1 --timeout 20"
        subprocess.run(cmd)  # noqa: S603, PLW1510
    # End do stuff

    # Finished
    stop_time = time.time()
    logger.info(
        "Node %s time spent in minutes: %s",
        ((rank - 1) % size),
        int(time.time() - start_time) / 60,
    )
