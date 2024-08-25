import geopandas as gpd


def load_dataset(split: str, post_training: int = 0) -> gpd.GeoDataFrame:
    """
    Loads data
    """
    if split == "all":
        return gpd.read_file("outputs/model/train-test-split.geojson")

    if split == "test" or (split == "train" and post_training == 0):
        dataset = gpd.read_file("outputs/model/train-test-split.geojson")
    elif split in ("train", "validation") and post_training == 1:
        dataset = gpd.read_file("outputs/model/train-validation-test-split.geojson")
    else:
        raise Exception(  # noqa: TRY002
            f"Unhandled dataset case: split = {split} and {'post training' if post_training else 'pre training'}")

    # Have to reset index here otherwise group split fails
    split_data = dataset[dataset["split"] == split].reset_index()

    return split_data
