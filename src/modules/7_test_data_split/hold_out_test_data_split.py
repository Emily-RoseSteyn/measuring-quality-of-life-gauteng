import random

import geopandas as gpd
from dvc.api import params_show

from utils.logger import get_logger
from utils.test_data_split import (
    test_data_split_ward_group_shuffle_split,
    test_data_split_simple_random,
)

logger = get_logger()

# Get DVC params
params = params_show()


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
