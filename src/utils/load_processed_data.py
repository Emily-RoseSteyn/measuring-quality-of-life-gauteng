import geopandas as gpd


def load_dataset(split: str):
    """
    Loads data
    """

    dataset = gpd.read_file("outputs/model/train-test-split.geojson")
    # Have to reset index here otherwise group split fails
    split_data = dataset[dataset["split"] == split].reset_index()
    return split_data
