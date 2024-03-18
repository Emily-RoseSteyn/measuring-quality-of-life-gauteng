#!/bin/bash
# Run from root dir of project
results_dir="${PWD}/outputs/tiles"

# Decompress results files
for file in "$results_dir"/result.*; do tar -xzf "${file}" -C "$results_dir" && rm "${file}"; done
echo "Uncompressed all files in output folder"

# When all workers done and outputs have been merged, merge geojson results only on the one node
python src/modules/tiling/merge_geojson "$results_dir"
echo "Merged geojson files in output folder"