import os
from pathlib import Path

import geopandas as gpd
from matplotlib import pyplot as plt

from utils.logger import get_logger


# NB: This is separate to tiles because we may want to match different clustered data with the same tiles
def main() -> None:
    logger = get_logger()
    logger.info("Matching tiles to clusters")

    # Output & tiles directory
    tiles_dir = os.path.abspath(Path("./outputs/tiles"))
    results_dir = os.path.abspath(Path("./outputs/matched"))

    if not os.path.isdir(results_dir):
        os.makedirs(results_dir)

    # Load clustered data
    qol_data = gpd.read_file("outputs/merged/gauteng-qol.geojson")

    # Get tile transforms
    tile_transforms = gpd.read_file(f"{tiles_dir}/tile-transforms.geojson")

    # Ensure same CRS
    tile_transforms = tile_transforms.to_crs(qol_data.crs)

    # Spatial join of data
    joined_data = tile_transforms.sjoin(qol_data, how="inner")

    # Plot
    joined_data.plot(column="qol_index", legend=True)
    plt.savefig(os.path.join(results_dir, "gauteng-qol-cluster-tiles.png"))

    # Save
    joined_data.to_file(
        os.path.join(results_dir, "gauteng-qol-cluster-tiles.geojson"), driver="GeoJSON"
    )


if __name__ == "__main__":
    main()
