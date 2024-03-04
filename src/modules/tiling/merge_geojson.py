import glob
import os

import geopandas as gpd
import pandas as pd


def merge_geojson(results_dir: str) -> None:
    pattern = os.path.join(results_dir, "*.geojson")
    file_list = glob.glob(pattern)

    collection = pd.DataFrame(columns=["tile", "geometry"])

    for file in file_list:
        geojson = gpd.read_file(file)
        # TODO: Fix warning about excluding empty collection
        collection = pd.concat([collection, geojson])
        os.remove(file)

    geo_collection = gpd.GeoDataFrame(collection)
    geo_collection.to_file(
        os.path.join(results_dir, "tile-transforms.geojson"),
        driver="GeoJSON",
        mode="w",
    )
