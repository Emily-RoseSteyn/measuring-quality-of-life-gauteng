import logging
import os

import geopandas as gpd
import matplotlib.pyplot as plt


def main() -> None:
    logging.info("In data download")
    # Load shapefile
    # TODO: Figure out best practice for path here
    #  this is executing relative to the root directory when using dvc repro
    shapefile = gpd.read_file("./data/shapefiles/2020/2020.shp")

    results_dir = "./outputs/intermediate/geojson"
    images_dir = "./outputs/intermediate/images/processed-shapefile"

    if not os.path.isdir(results_dir):
        os.makedirs(results_dir)

    if not os.path.isdir(images_dir):
        os.makedirs(images_dir)

    # Save intermediate shapefile - all wards in Gauteng
    gauteng_wards = shapefile[shapefile.Province == "Gauteng"]
    gauteng_wards.plot()
    plt.savefig(os.path.join(images_dir, "gauteng-wards-shapefile.png"))
    gauteng_wards.to_file(
        os.path.join(results_dir, "gauteng-wards.geojson"), driver="GeoJSON"
    )

    # Save intermediate shapefile - Gauteng boundary
    gauteng_boundary = gauteng_wards[["Province", "geometry"]]
    gauteng_boundary = gauteng_boundary.dissolve(by="Province")
    gauteng_boundary.plot(facecolor="none")

    plt.savefig(os.path.join(images_dir, "gauteng-boundary-shapefile.png"))
    gauteng_boundary.to_file(
        os.path.join(results_dir, "gauteng-boundary.geojson"), driver="GeoJSON"
    )

    # Load gcro data
    # TODO: Analyse dates of GCRO data
    # Save selected dates
    # Make request to basemaps api


if __name__ == "__main__":
    main()
