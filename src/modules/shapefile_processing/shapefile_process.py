import logging
import os

import geopandas as gpd
import matplotlib.pyplot as plt


def main() -> None:
    logging.info("In shapefile processing")
    # Load shapefile
    # TODO: Figure out best practice for path here
    #  this is executing relative to the root directory when using dvc repro
    shapefile = gpd.read_file("./data/shapefiles/2020/2020.shp")

    # Replace with step parameter
    results_dir = "./outputs/processed_shapefile"

    if not os.path.isdir(results_dir):
        os.makedirs(results_dir)

    # Save intermediate shapefile - all wards in Gauteng
    gauteng_wards = shapefile[shapefile.Province == "Gauteng"]
    gauteng_wards.plot()
    plt.savefig(os.path.join(results_dir, "gauteng-wards.png"))
    gauteng_wards.to_file(
        os.path.join(results_dir, "gauteng-wards.geojson"), driver="GeoJSON"
    )

    # Save intermediate shapefile - Gauteng boundary
    gauteng_boundary = gauteng_wards[["Province", "geometry"]]
    gauteng_boundary = gauteng_boundary.dissolve(by="Province")
    # This gives some artifacts in the middle because of non-contiguous lines supposedly
    gauteng_boundary.plot(facecolor="none")
    plt.savefig(os.path.join(results_dir, "gauteng-boundary-unprocessed.png"))
    gauteng_boundary.to_file(
        os.path.join(results_dir, "gauteng-boundary-unprocessed.geojson"),
        driver="GeoJSON",
    )

    # Process with buffer so that artifacts are removed
    gauteng_processed_boundary = gauteng_boundary.buffer(0.001)
    gauteng_processed_boundary.plot(facecolor="none")
    plt.savefig(os.path.join(results_dir, "gauteng-boundary.png"))
    gauteng_processed_boundary.to_file(
        os.path.join(results_dir, "gauteng-boundary.geojson"),
        driver="GeoJSON",
    )


if __name__ == "__main__":
    main()
