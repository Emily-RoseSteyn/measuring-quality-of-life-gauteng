import random

import geopandas as gpd
import pandas as pd
from sklearn.model_selection import train_test_split

from utils.logger import get_logger

logger = get_logger()


def split_data(df: pd.DataFrame) -> tuple:
    """
    Accepts a Pandas DataFrame and splits it into training and test data. Returns DataFrames.

    Parameters
    ----------
    df : pd.DataFrame
        Pandas DataFrame containing all data.

    Returns
    -------
    Union[pd.DataFrame, pd.DataFrame, pd.DataFrame]
        [description]
    """
    # TODO: Splits should be params; also note eating into training data here
    # TODO: Ensure splitting by ward + distribution of some kind??
    # TODO: Distribution of these datasets needs to be fair?
    #  See e2e notebooks with stratified shuffle split
    train, test = train_test_split(df, test_size=0.2, random_state=1)  # split the data with a validation size o 20%

    logger.info("Descriptive statistics of train:")
    logger.info(f"Shape: {train.shape}")
    logger.info(train.describe())

    logger.info("Descriptive statistics of test:")
    logger.info(f"Shape: {test.shape}")
    logger.info(test.describe())

    df.loc[train.index, "split"] = "train"
    df.loc[test.index, "split"] = "test"
    df.to_csv("outputs/model/train-test-split.csv")

    return train, test


def main() -> None:
    logger.info("In test data split")

    # Seeding
    seed = 42
    random.seed(seed)
    # np.random.seed(seed)  # TODO: Figure out how to fix this

    # Load dataset
    dataset = gpd.read_file("outputs/matched/gauteng-qol-cluster-tiles.geojson")

    # Split data into 8_training, validation, test datasets
    split_data(dataset)


if __name__ == "__main__":
    main()
