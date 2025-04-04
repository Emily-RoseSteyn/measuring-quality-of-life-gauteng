import os

import geopandas as gpd
import matplotlib.pyplot as plt

from utils.logger import get_logger


def process_shapefile_by_date(date: str):
    # Load shapefile
    # TODO: Figure out best practice for path here
    #  this is executing relative to the root directory when using dvc repro
    shapefile = gpd.read_file(f"./data/shapefiles/{date}/{date}.shp")
    shapefile = shapefile.rename(
        columns={
            "ProvinceNa": "Province",
        }
    )

    # TODO: Replace with step parameter
    results_dir = f"./outputs/processed-shapefile/{date}"

    if not os.path.isdir(results_dir):
        os.makedirs(results_dir)

    # Save image of all wards in SA
    shapefile.plot()
    plt.savefig(os.path.join(results_dir, "za-wards.png"))

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
    # Change projection so that buffer applied as expected
    # From geodetic coordinates (EPSG4326) to meters (3857) and back again
    gauteng_processed_boundary = gauteng_boundary.to_crs(crs=3857)
    gauteng_processed_boundary["geometry"] = (
        gauteng_processed_boundary["geometry"].buffer(0.01).buffer(-0.01)
    )
    gauteng_processed_boundary = gauteng_processed_boundary.to_crs(
        crs=gauteng_boundary.crs
    )

    gauteng_processed_boundary.plot(facecolor="none")
    plt.savefig(os.path.join(results_dir, "gauteng-boundary.png"))
    gauteng_processed_boundary.to_file(
        os.path.join(results_dir, "gauteng-boundary.geojson"),
        driver="GeoJSON",
    )


def main() -> None:
    logger = get_logger()
    logger.info("In shapefile processing")

    process_shapefile_by_date("2011")
    process_shapefile_by_date("2016")
    process_shapefile_by_date("2020")


if __name__ == "__main__":
    main()
