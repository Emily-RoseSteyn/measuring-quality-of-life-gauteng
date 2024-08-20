import geopandas as gpd
import pandas as pd
from dvc.api import params_show

params = params_show()["preprocessing"]


def select_year(dataset: gpd.GeoDataFrame):
    year = params["year"]

    assert year in ["all", "2018", "2021"]

    if year == "all":
        return dataset

    return dataset[dataset["year"] == year]


def select_data(dataset: gpd.GeoDataFrame) -> pd.DataFrame:
    """
    Given various conditions, select sections of data
    """

    # Select year
    dataset = select_year(dataset)
    return dataset.reset_index()
