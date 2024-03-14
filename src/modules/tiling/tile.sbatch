#!/bin/bash
#SBATCH --job-name=tiling_emily
#SBATCH --output=logs/%x-%A_%a.out.log
#SBATCH --error=logs/%x-%A_%a.err.log
#SBATCH --ntasks=20
#SBATCH --cpus-per-task=1

# Python unbuffered for logs
export PYTHONUNBUFFERED=TRUE

# Load .env file
source .env

# Access the variable
dir=$SCRATCH

# Create temp directory required for all nodes
exec srun mkdir -p $dir
wait

# Assuming relative to root project directory
exec srun --unbuffered --mpi=pmi2 python ./src/modules/tiling/tile_slurm.py
wait

# Remove temp directory required for all nodes
exec srun rm -rf $dir
wait