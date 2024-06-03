import random

import geopandas as gpd
from dvc.api import params_show
from sklearn.model_selection import GroupShuffleSplit

from utils.logger import get_logger

logger = get_logger()

# Get DVC params
params = params_show()


def ward_test_data_split(df: gpd.GeoDataFrame) -> tuple:
    """
    Accepts a Pandas DataFrame and splits it into training and test data. Saves these.

    Parameters
    ----------
    df : pd.DataFrame
        Pandas DataFrame containing all data.

    Returns
    -------
    Union[pd.DataFrame, pd.DataFrame]
    """
    # Split data
    random_state = params["constants"]["random_seed"]
    test_size = params["split"]["test_size"]

    # NB - test size here is for the group (ie 20% of wards will go into group. Not 20% of tiles)
    gss = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=random_state)
    groups = df["ward_code"]

    # Get the next item from the gss iterator. Only expecting one because of n_splits=1 above
    # Split function returns indexes in group!
    train_index, test_index = next(gss.split(df, groups=groups))

    # Access train and test grouped data
    train_groups = groups[train_index]
    test_groups = groups[test_index]
    logger.info("Ward information for train and test")
    logger.info(f"Wards in train: {len(train_groups.unique())}")
    logger.info(f"Wards in test: {len(test_groups.unique())}")

    train = df.loc[train_groups.index]
    logger.info("Descriptive statistics of train:")
    logger.info(f"Shape: {train.shape}")
    logger.info(train.describe())

    test = df.loc[test_groups.index]
    logger.info("Descriptive statistics of test:")
    logger.info(f"Shape: {test.shape}")
    logger.info(test.describe())

    df.loc[train.index, "split"] = "train"
    df.loc[test.index, "split"] = "test"
    df.to_file("outputs/model/train-test-split.geojson", driver="GeoJSON")
    return train, test


def main() -> None:
    logger.info("In test data split")

    # Seeding
    seed = params["constants"]["random_seed"]
    random.seed(seed)

    # Load dataset
    dataset = gpd.read_file("outputs/matched/gauteng-qol-cluster-tiles.geojson")

    # If simple random

    # If ward random

    # Split data into train and test datasets
    ward_test_data_split(dataset)


if __name__ == "__main__":
    main()
