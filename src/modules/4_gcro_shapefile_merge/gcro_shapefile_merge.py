import os
from time import strptime

import geopandas as gpd
import pandas as pd
from matplotlib import pyplot as plt
from numpy import sort

from gcro_shapefile_map import GCRO_SHAPEFILE_MAP
from utils.logger import get_logger

logger = get_logger()


def find_shapefile_year_dynamically(gcro_year: str):
    shapefile_output = "outputs/processed-shapefile"
    shapefile_years = sort([f.name for f in os.scandir(shapefile_output) if f.is_dir()])

    selected_shapefile_year = shapefile_years[0]
    # Assuming sorted
    for year in shapefile_years:
        gcro_year_tuple = strptime(gcro_year, "%Y")
        shapefile_year_tuple = strptime(year, "%Y")

        if gcro_year_tuple >= shapefile_year_tuple:
            selected_shapefile_year = year

    return selected_shapefile_year


def merge_gcro_shapefile_by_gcro_year(year: str):
    logger.info(f"Merging {year}")
    results_dir = f"./outputs/merged/{year}"
    if not os.path.isdir(results_dir):
        os.makedirs(results_dir)

    # Load qol data and ensure ward code column matches type of geojson (string)
    data_path = f"outputs/processed-gcro/{year}/gcro-clustered-data.csv"
    qol_data = pd.read_csv(data_path)
    qol_data["ward_code"] = qol_data["ward_code"].astype(str)

    # Load geojson data and select specific columns
    shapefile_year = GCRO_SHAPEFILE_MAP[year]
    logger.info(f"Found matching shapefile {shapefile_year}")
    geojson_path = f"outputs/processed-shapefile/{shapefile_year}/gauteng-wards.geojson"
    gdf = gpd.read_file(geojson_path)
    gdf = gdf[["WardID", "geometry"]].rename(columns={"WardID": "ward_code"})

    # Check for missing wards between the two datasets
    missing = gdf.merge(qol_data, how="outer", on="ward_code", indicator=True).query(
        '_merge=="left_only"'
    )

    if len(missing) > 0:
        raise Exception("Mismatch between the GCRO data and shapefile")  # noqa: TRY002

    # Merge the two datasets on ward code
    overlap = gdf.merge(qol_data, how="inner", on="ward_code")

    # Plot
    overlap.plot(column="qol_index", legend=True)
    plt.savefig(os.path.join(results_dir, "gauteng-qol.png"))
    overlap.to_file(os.path.join(results_dir, "gauteng-qol.geojson"), driver="GeoJSON")


def main() -> None:
    logger.info("In gcro shapefile merge")

    # Merge each individual gcro and shapefile
    gcro_output = "outputs/processed-gcro"
    gcro_years = sort([f.name for f in os.scandir(gcro_output) if f.is_dir()])
    for year in gcro_years:
        merge_gcro_shapefile_by_gcro_year(year)

    # Merge all data together


if __name__ == "__main__":
    main()
