import logging
import os

import geopandas as gpd
import numpy as np
from geopandas import GeoDataFrame
from plotnine import (
    aes,
    geom_map,
    ggplot,
    labs,
    scale_fill_cmap,
)
from shapely import box


def create_grid(bounds: np.ndarray, cell_size: float) -> GeoDataFrame:
    x_min, y_min, x_max, y_max = bounds
    cols = np.arange(x_min, x_max + cell_size, cell_size)
    rows = np.arange(y_min, y_max + cell_size, cell_size)
    grid_cells = []
    for x0 in cols:
        for y0 in rows:
            # bounds
            x1 = x0 - cell_size
            y1 = y0 + cell_size
            grid_cells.append(box(x0, y0, x1, y1))
    return gpd.GeoDataFrame(grid_cells, columns=["geometry"])


def main() -> None:
    logging.info("In grid generation")

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
    grid = create_grid(
        bbox,
        cell_size=target_resolution,
    )

    # Plot grid over shp with values
    # noinspection PyTypeChecker
    #   - Warning is incorrect
    fig = (
        ggplot()
        + geom_map(data=qol_labels, mapping=aes(fill="qol_index"))
        + scale_fill_cmap()
        + geom_map(data=grid, fill=None)
        + labs(x="", y="", fill="QoL Index")
    )
    fig.save(f"{results_dir}/grid-overlay.png", dpi=300)


if __name__ == "__main__":
    main()
