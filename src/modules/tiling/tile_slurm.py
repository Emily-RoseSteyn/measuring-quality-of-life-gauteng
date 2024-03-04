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
    start_time = time.time()
    comm = MPI.COMM_WORLD  # get MPI communicator object
    rank = comm.Get_rank()
    size = comm.Get_size()
    num_workers = size - 1
    name = MPI.Get_processor_name()
    status = MPI.Status()  # get MPI status object

    # If rank 0, listen for all workers to be done
    if rank == 0:
        logger.info(f"Master starting with {num_workers:d} workers")
        closed_workers = 0
        task_index = 0

        # This is basically a poll for receiving comms
        while closed_workers < num_workers:
            comm.recv(source=MPI.ANY_SOURCE, tag=MPI.ANY_TAG, status=status)
            source = status.Get_source()
            tag = status.Get_tag()
            if tag == MPI_TAGS.READY:
                # Worker is ready, so send it a task
                if task_index < len(file_list):
                    comm.send({"file": file_list[task_index], "index": task_index}, dest=source, tag=MPI_TAGS.START)
                    logger.debug(f"Sending task {task_index} to worker {source}")
                    task_index += 1
                else:
                    comm.send(None, dest=source, tag=MPI_TAGS.EXIT)
            elif tag == MPI_TAGS.DONE:
                logger.debug(f"Got data from worker {source}")
            elif tag == MPI_TAGS.EXIT:
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
        logger.info(f"Tiling with slurm - node {rank}")

        # Worker processes execute code below
        name = MPI.Get_processor_name()
        while True:
            comm.send(None, dest=0, tag=MPI_TAGS.READY)
            task = comm.recv(source=0, tag=MPI.ANY_TAG, status=status)
            tag = status.Get_tag()

            if tag == MPI_TAGS.START:
                # Do the work here
                item = task["file"]
                index = task["index"]
                logger.info(
                    "Item %s being done by processor %d (%s) of %d", item, rank, name, num_workers
                )
                tile_image(item, output_directory, thread=index)
                comm.send(None, dest=0, tag=MPI_TAGS.DONE)
            elif tag == MPI_TAGS.EXIT:
                break

        # When node done with its file list, tell rank 0
        comm.send(None, dest=0, tag=MPI_TAGS.EXIT)

        # Finished
        logger.info(
            "Node %s time spent in minutes: %s",
            rank,
            int(time.time() - start_time) / 60,
        )


def main() -> None:
    images = get_tile_list()
    results_dir = os.path.abspath(Path("./outputs/tiles"))
    tile_slurm(images, results_dir)


if __name__ == "__main__":
    main()
