import geopandas as gpd
import pandas as pd
from dvc.api import params_show

params = params_show()
preprocessing_params = params["preprocessing"]


# Need more than k-fold wards => k-folds for training set + at least 1 more for test
MIN_CUSTOM_WARDS = params["split"]["folds"]

# def select_tiles(dataset: gpd.GeoDataFrame):
#     # If drop overlap, remove any duplicate tiles
#     drop_overlap = preprocessing_params["drop_overlap"]
#
#     if drop_overlap:
#         grouped_by_tile = dataset.groupby("tile").count()
#         multi_ward_tiles = grouped_by_tile[grouped_by_tile["year"] > 1].reset_index()
#         dataset = dataset[~dataset["tile"].isin(multi_ward_tiles["tile"])]
#     return dataset


def select_year(dataset: gpd.GeoDataFrame):
    year = preprocessing_params["year"]

    assert year in ["all", "2018", "2021"]

    if year == "all":
        return dataset

    return dataset[dataset["year"] == year]


def select_wards(dataset: gpd.GeoDataFrame):
    custom_wards = preprocessing_params["custom_wards"]
    if custom_wards:
        # Get custom wards and current wards
        wards = pd.read_csv("data/custom/train_wards.csv", dtype=str)
        current_ward_codes = dataset["ward_code"].unique()

        # Check if sufficient wards match
        wards_match = wards.isin(current_ward_codes)
        wards_match = wards_match[wards_match.ward_code]
        if len(wards_match) <= MIN_CUSTOM_WARDS:
            raise Exception("Too few valid wards in custom ward data")  # noqa: TRY002

        return dataset[dataset["ward_code"].isin(wards["ward_code"])]

    return dataset


def select_data(dataset: gpd.GeoDataFrame) -> pd.DataFrame:
    """
    Given various conditions, select sections of data
    """
    # TODO: Select non-overlapping tiles if param set
    # dataset = select_tiles(dataset)

    # Select year
    dataset = select_year(dataset)

    # Select wards
    dataset = select_wards(dataset)

    # TODO: Plot data again
    return dataset.reset_index()
