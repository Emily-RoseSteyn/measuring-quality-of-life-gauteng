import os

import geopandas as gpd
import numpy as np
from geopandas import GeoDataFrame
from plotnine import (
    aes,
    geom_map,
    ggplot,
    labs,
    scale_color_cmap,
    scale_fill_cmap,
)
from utils.logger import get_logger


def create_grid(bounds: np.ndarray, cell_size: float, crs: str) -> GeoDataFrame:
    x_min, y_min, x_max, y_max = bounds

    # Offsetting to correspond to MOSAIKS grid precision
    x_min = round(x_min, 2) + cell_size / 2
    y_min = round(y_min) + cell_size / 2
    x_max = round(x_max, 2) + cell_size / 2
    y_max = round(y_max, 2) + cell_size / 2

    # Get list of point coordinates
    # NB: Previously used np.arange without rounding, but this has precision errors for decimal cell_size
    x_coords = list(np.around(np.arange(x_min, x_max, cell_size), 3))
    y_coords = list(np.around(np.arange(y_min, y_max, cell_size), 3))

    # Create all combinations of xy coordinates
    coordinate_pairs = np.array(np.meshgrid(x_coords, y_coords)).T.reshape(-1, 2)

    # Create a list of shapely points
    geometries = gpd.points_from_xy(coordinate_pairs[:, 0], coordinate_pairs[:, 1])

    return gpd.GeoDataFrame(geometries, columns=["geometry"], crs=crs)


def main() -> None:
    logger = get_logger()
    logger.info("In grid generation")

    results_dir = "./outputs/grid"
    if not os.path.isdir(results_dir):
        os.makedirs(results_dir)

    # Load shapefile
    qol_labels = gpd.read_file("outputs/merged/gauteng-qol.geojson")

    # Get bounding box
    bbox = qol_labels.total_bounds

    # Set target resolution. A resolution of .01 is equal to the .01 x .01 dense grid of the MOSAIKS API.
    target_resolution = 0.01

    # Creates grid of at specified resolution starting at the
    # min x and min y coordinates of the label bounding box
    # Rounded to 3 digit precision of the MOSAIKS grid
    grid = create_grid(bbox, cell_size=target_resolution, crs=qol_labels.crs)

    # Plot grid over shp with values
    # noinspection PyTypeChecker
    #   - Warning is incorrect
    fig = (
        ggplot()
        + geom_map(data=qol_labels, mapping=aes(fill="qol_index"))
        + scale_fill_cmap(cmap_name="plasma")
        + geom_map(data=grid, fill=None)
        + labs(x="", y="", fill="QoL Index")
    )
    fig.save(f"{results_dir}/grid-overlay.png", dpi=300)

    # Spatial join grid over shapefile
    joined_gdf = gpd.sjoin(grid, qol_labels, how="inner", predicate="within")

    # Plot joined grid
    # noinspection PyTypeChecker
    #   - Warning is incorrect
    fig = (
        ggplot()
        + geom_map(data=joined_gdf, mapping=aes(color="qol_index"), size=0.01)
        + scale_color_cmap(cmap_name="plasma")
        + labs(x="", y="", color="Quality of Life")
    )
    fig.save(f"{results_dir}/grid-gauteng-qol.png", dpi=300)

    # Add latitude and longitude
    joined_gdf["longitude"] = joined_gdf.geometry.x
    joined_gdf["latitude"] = joined_gdf.geometry.y

    # Save CSV
    joined_gdf.to_csv(f"{results_dir}/qol-labelled-grid.csv")


if __name__ == "__main__":
    main()
