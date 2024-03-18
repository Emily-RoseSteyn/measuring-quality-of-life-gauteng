#!/bin/bash
#SBATCH --job-name=tiling_emily
#SBATCH --output=logs/%x-%A_%a.out.log
#SBATCH --error=logs/%x-%A_%a.err.log
#SBATCH --ntasks=20
#SBATCH --cpus-per-task=1

echo "Starting tiling with SBATCH"

# Python unbuffered for logs
export PYTHONUNBUFFERED=TRUE

# Load .env file
source .env

# Access the variable
dir=$SCRATCH

# Create temp directory required for all nodes
srun mkdir -p "$dir"
wait
echo "Made temp directories on nodes"

# Assuming relative to root project directory
srun --unbuffered --mpi=pmi2 python src/modules/tiling/tile_slurm.py
wait

echo "Completed tiling"

# Move outputs from temp dir to outputs in working dir
# TODO: directory should be in env
results_dir="${SLURM_SUBMIT_DIR}/outputs/tiles"

# Gather from nodes (under the hood uses scp)
# Move is to outputs/tiles.<NODE_NAME>
sgather -r -v -C "$dir" "$results_dir"
wait
echo "Moved tiles from nodes to output folder"

# Merge tiles folders

# When all workers done and outputs have been merged, merge geojson results only on the one node
#python src/modules/tiling/merge_geojson "$results_dir"
#echo "Merged geojson files in output folder"

srun rm -rf "$dir"
echo "Done"
