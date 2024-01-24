import os

import geopandas as gpd
import pandas as pd
from matplotlib import pyplot as plt
from utils.logger import get_logger


def main() -> None:
    logger = get_logger()
    logger.info("In gcro shapefile merge")

    results_dir = "./outputs/merged"
    if not os.path.isdir(results_dir):
        os.makedirs(results_dir)

    # Load geojson data and select specific columns
    geojson_path = "outputs/processed-shapefile/gauteng-wards.geojson"
    gdf = gpd.read_file(geojson_path)
    gdf = gdf[["WardID", "geometry"]].rename(columns={"WardID": "ward_code"})

    # Load qol data and ensure ward code column matches type of geojson (string)
    data_path = "outputs/processed-gcro/gcro-clustered-data.csv"
    qol_data = pd.read_csv(data_path)
    qol_data["ward_code"] = qol_data["ward_code"].astype(str)

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


if __name__ == "__main__":
    main()
