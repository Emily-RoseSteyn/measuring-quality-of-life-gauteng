#!/bin/bash
#SBATCH --job-name=tiling_emily
#SBATCH --output=~/logs/%x-%A_%a.out.log
#SBATCH --error=~/logs/%x-%A_%a.err.log
#SBATCH --cpus-per-task 1
#SBATCH --ntasks=25

echo ${PWD}

exec srun --unbuffered --mpi=pmi2 tile_slurm.py
