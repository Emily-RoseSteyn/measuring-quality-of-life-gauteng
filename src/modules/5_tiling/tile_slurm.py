import os
import time
from pathlib import Path

from mpi4py import MPI

from get_tile_list import get_tile_list
from tile_image import tile_image
from utils.env_variables import MPI_TAGS, TEMP_WRITE_DIR
from utils.logger import get_logger

logger = get_logger()


def root_process(comm, num_workers, status):
    logger.info(f"Master starting with {num_workers:d} workers")
    file_list = get_tile_list()
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


def worker_process(comm, num_workers, rank, status):
    logger.info(f"Tiling with slurm - node {rank}")

    # Setup output dir
    output_dir = os.path.abspath(Path(TEMP_WRITE_DIR))
    if not os.path.isdir(output_dir):
        os.makedirs(output_dir)

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
            tile_image(item, output_dir, thread=index)
            comm.send(None, dest=0, tag=MPI_TAGS.DONE)
        elif tag == MPI_TAGS.EXIT:
            break

    # When node done with its file list, tell rank 0
    comm.send(None, dest=0, tag=MPI_TAGS.EXIT)


def main() -> None:
    start_time = time.time()
    comm = MPI.COMM_WORLD  # get MPI communicator object
    rank = comm.Get_rank()
    size = comm.Get_size()
    num_workers = size - 1
    MPI.Get_processor_name()
    status = MPI.Status()  # get MPI status object

    # If rank 0, listen for all workers to be done
    if rank == 0:
        root_process(comm, num_workers, status)

    # If not rank 0, then worker node
    else:
        worker_process(comm, num_workers, rank, status)

    # Finished
    logger.info(
        "Node %s time spent in minutes: %s",
        rank,
        int(time.time() - start_time) / 60,
    )


if __name__ == "__main__":
    main()
