import os
import time
from pathlib import Path

from mpi4py import MPI

from modules.tiling.get_tile_list import get_tile_list
from modules.tiling.merge_geojson import merge_geojson
from modules.tiling.tile_image import tile_image
from utils.env_variables import MPI_TAGS
from utils.logger import get_logger

logger = get_logger()


def tile_slurm(file_list: list, output_directory: str) -> None:
    logger.info("Tiling with slurm")

    start_time = time.time()
    comm = MPI.COMM_WORLD  # get MPI communicator object
    rank = comm.Get_rank()
    size = comm.Get_size()
    name = MPI.Get_processor_name()
    status = MPI.Status()  # get MPI status object

    # If rank 0, listen for all workers to be done
    if rank == 0:
        num_workers = size - 1
        logger.info(f"Master starting with {num_workers:d} workers")
        closed_workers = 0

        # Make sure rank 0 has gotten to this point before moving on
        MPI.COMM_WORLD.Barrier()

        # This is basically a poll for receiving comms
        while closed_workers < num_workers:
            source = status.Get_source()
            tag = status.Get_tag()

            # Listen for comms from workers
            comm.recv(source=MPI.ANY_SOURCE, tag=MPI.ANY_TAG, status=status)
            logger.info(f"Got data from worker {source}")

            if tag == MPI_TAGS.EXIT:
                logger.info(f"Worker {source} exited.")
                closed_workers += 1

        logger.info("All workers finished")

        # When all workers done, merge geojson results
        merge_geojson(output_directory)

        logger.info(
            "Master node done. Time spent in minutes: %s",
            int(time.time() - start_time) / 60,
        )

    # If not rank 0, then worker node
    else:
        # For each file
        for index, item in enumerate(file_list):
            # If there are still files to process and file is modulus of current rank
            # (means that every node processing files independently)
            if index % size != rank:
                continue

            logger.info(
                "Item %s being done by processor %d (%s) of %d", item, rank, name, size
            )

            # Do stuff here!
            tile_image(item, output_directory, thread=index)

        # When node done with its file list, tell rank 0
        comm.send(None, dest=0, tag=MPI_TAGS.EXIT)
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
