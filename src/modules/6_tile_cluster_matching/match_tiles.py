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
    # Assumes that geometry and year are keys in both qol_data and tile_transforms
    joined_data = tile_transforms.sjoin(qol_data, how="inner")

    # Select only the rows with matching years
    joined_data = joined_data.loc[joined_data["year_left"] == joined_data["year_right"]]
    joined_data = joined_data.rename(columns={"year_left": "year"})
    joined_data = joined_data.drop(["year_right", "index_right"], axis=1)

    # Plot
    joined_data[joined_data["year"] == "2018"].plot(column="qol_index", legend=True, aspect=1)
    plt.savefig(os.path.join(results_dir, "2018-gauteng-qol-cluster-tiles.png"))
    joined_data[joined_data["year"] == "2021"].plot(column="qol_index", legend=True, aspect=1)
    plt.savefig(os.path.join(results_dir, "2021-gauteng-qol-cluster-tiles.png"))

    # Save
    joined_data.to_file(
        os.path.join(results_dir, "gauteng-qol-cluster-tiles.geojson"), driver="GeoJSON"
    )


if __name__ == "__main__":
    main()
