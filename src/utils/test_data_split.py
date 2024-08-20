import pandas as pd
from dvc.api import params_show
from sklearn.model_selection import GroupShuffleSplit, train_test_split

params = params_show()


def test_data_split_ward_group_shuffle_split(
    df: pd.DataFrame, test_size=params["split"]["test_size"]
) -> tuple:
    """
    Accepts a DataFrame and splits it into training and test data
    grouped by ward so that no ward overlaps between the train/test dataset.

    Parameters
    ----------
    df : pd.DataFrame
        Pandas DataFrame containing all data.
    test_size:
        The test size split - defaults to test size from params for actual hold out test set

    Returns
    -------
    Union[pd.DataFrame, pd.DataFrame]
    """
    # Split data
    random_state = params["constants"]["random_seed"]

    # NB - test size here is for the group (eg 20% of wards will go into group. Not 20% of tiles)
    gss = GroupShuffleSplit(test_size=test_size, random_state=random_state)
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


def test_data_split_simple_random(
    df: pd.DataFrame, test_size=params["split"]["test_size"]
) -> tuple:
    """
    Accepts a DataFrame and splits it into training and test data randomly.

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
    return train_test_split(df, test_size=test_size, random_state=random_state)
