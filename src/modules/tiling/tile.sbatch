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

# Set output dir variable
dir=$SCRATCH

# Create temp directory required for all nodes
srun mkdir -p "$dir"
wait
echo "Made temp directories on nodes"

# Assuming relative to root project directory
srun --unbuffered --mpi=pmi2 python src/modules/tiling/tile_slurm.py
wait
echo "Completed tiling"

# Compress output directory but only run once on each node
# I do this compression because moving many small files with sgather is slow (because it uses scp under the hood)
# Alternatively could do rsync but leaving as is for now
number_nodes=$SLURM_JOB_NUM_NODES
compressed_dir="$dir".tar.gz
srun --ntasks="$number_nodes" --ntasks-per-node=1 tar -zcf "$compressed_dir" "$dir"
wait
echo "Compressed tiled directories"

# Move outputs from temp dir to outputs in working dir
# TODO: directory should be in env
results_dir="${SLURM_SUBMIT_DIR}/outputs/tiles"

# Gather from nodes (under the hood uses scp)
# Move is to outputs/tiles/result.<NODE_NAME>
sgather -vC "$compressed_dir" "$results_dir/result"
wait
echo "Moved tiles from nodes to output folder"

# Remove temp directory on nodes
srun --ntasks="$number_nodes" --ntasks-per-node=1 rm -rf "$dir"
wait
echo "Removed temp directory on nodes"

# Decompress results files
for file in "$results_dir"/result.*; do tar xzvf "${file}" && rm "${file}"; done
echo "Uncompressed all files in output folder"

# When all workers done and outputs have been merged, merge geojson results only on the one node
python src/modules/tiling/merge_geojson "$results_dir"
echo "Merged geojson files in output folder"

echo "Done"
