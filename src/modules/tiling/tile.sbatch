#!/bin/bash
#SBATCH --job-name=tiling_emily
#SBATCH --output=logs/%x-%A_%a.out.log
#SBATCH --error=logs/%x-%A_%a.err.log
#SBATCH --cpus-per-task 4
#SBATCH --ntasks=4

# Python unbuffered for logs
export PYTHONUNBUFFERED=TRUE

# Assuming relative to root project directory
exec srun --unbuffered --mpi=pmi2 python ./src/modules/tiling/tile_slurm.py
