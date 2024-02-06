#!/bin/bash
#SBATCH --job-name=tiling_emily
#SBATCH --output=/home/esteyn/logs/%x-%A_%a.out.log
#SBATCH --error=/home/esteyn/logs/%x-%A_%a.err.log
#SBATCH --cpus-per-task 1
#SBATCH --ntasks=25

# Note to Emily - there are 20 compute nodes each with 4 cores and 32GB mem
# ntasks above is the number of cores to use - 40 is half the cluster, 80 is full
# if running long jobs rather use 40 so that others can still use the system
# You can also get it to email you and a whole bunch of other things, but not necessary for now

exec srun --unbuffered --mpi=pmi2 python -c "print('hello')"
# TODO: Correct command to point to tile_slurm.py