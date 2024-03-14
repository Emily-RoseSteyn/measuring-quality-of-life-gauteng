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
srun mkdir -p $dir
wait

# Assuming relative to root project directory
srun --unbuffered --mpi=pmi2 python src/modules/tiling/tile_slurm.py
wait

# Move outputs from temp dir to outputs in working dir
echo $PWD
$results_dir=output/tiles
mv $dir/* $results_dir

# When all workers done and outputs have been merged, merge geojson results
python src/modules/tiling/merge_geojson $results_dir

srun rm -rf $dir
