import os
from pathlib import Path

import geopandas as gpd
from utils.logger import get_logger


# NB: This is separate to tiles because we may want to match different clustered data with the same tiles
def main() -> None:
    logger = get_logger()
    logger.info("Matching tiles to clusters")

    # Output & tiles directory
    results_dir = os.path.abspath(Path("./outputs/tiles"))

    # Get tile transforms
    tile_transforms = gpd.read_file(f"{results_dir}/tile-transforms.geojson")

    # Load clustered data
    qol_data = gpd.read_file("outputs/merged/gauteng-qol.geojson")

    # Ensure same CRS
    qol_data = qol_data.to_crs(tile_transforms.crs)

    # TODO: Validate this
    intersections = gpd.overlay(qol_data, tile_transforms, how="intersection")

    # Save
    intersections.to_file(
        os.path.join(results_dir, "gauteng-qol-cluster-tiles.geojson"), driver="GeoJSON"
    )


if __name__ == "__main__":
    main()
