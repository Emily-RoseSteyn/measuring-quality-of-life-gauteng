import random

import geopandas as gpd
from dvc.api import params_show
from sklearn.model_selection import GroupShuffleSplit, train_test_split

from utils.logger import get_logger

logger = get_logger()

# Get DVC params
params = params_show()


def test_data_split_simple_random(df: gpd.GeoDataFrame) -> tuple:
    """
    Accepts a Geopandas DataFrame and splits it into training and test data randomly.

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
    return train_test_split(df, test_size=test_size, random_state=random_state)


def test_data_split_ward_group_shuffle_split(df: gpd.GeoDataFrame) -> tuple:
    """
    Accepts a Geopandas DataFrame and splits it into training and test data
    grouped by ward so that no ward overlaps between the train/test dataset.

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

    train = df.loc[train_groups.index]
    test = df.loc[test_groups.index]

    return train, test


def main() -> None:
    logger.info("In test data split")

    # Seeding
    seed = params["constants"]["random_seed"]
    random.seed(seed)

    # Load dataset
    dataset = gpd.read_file("outputs/matched/gauteng-qol-cluster-tiles.geojson")

    # Split data into train and test datasets
    # Type of dataset split
    group_by_ward = params["split"]["group_by_ward"]

    # If group by ward
    if group_by_ward:
        logger.info("Grouping by ward")
        train, test = test_data_split_ward_group_shuffle_split(dataset)
    # Else assume simple random
    else:
        logger.info("Random splitting")
        train, test = test_data_split_simple_random(dataset)

    logger.info("Ward information for train and test")
    train_wards = train["ward_code"].unique()
    test_wards = test["ward_code"].unique()
    logger.info(f"Wards in train: {len(train_wards)}")
    logger.info(f"Wards in test: {len(test_wards)}")

    logger.info("Descriptive statistics of train:")
    logger.info(f"Shape: {train.shape}")
    train.describe().to_csv("outputs/model/train-stats.csv")

    logger.info("Descriptive statistics of test:")
    logger.info(f"Shape: {test.shape}")
    test.describe().to_csv("outputs/model/test-stats.csv")

    dataset.loc[train.index, "split"] = "train"
    dataset.loc[test.index, "split"] = "test"
    dataset.to_file("outputs/model/train-test-split.geojson", driver="GeoJSON")

    # TODO: Might want to add a plot of train vs test?


if __name__ == "__main__":
    main()
