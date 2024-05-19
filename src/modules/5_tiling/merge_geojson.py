import argparse
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


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    results_dir_key = "results_dir"
    parser.add_argument(f"{results_dir_key}", help="The results directory to merge geojson")
    args = vars(parser.parse_args())
    results_dir = args[results_dir_key]

    merge_geojson(results_dir)
