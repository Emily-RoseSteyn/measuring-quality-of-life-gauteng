import tensorflow as tf
from pandas import merge, read_csv
from utils.logger import get_logger


def main() -> None:
    logger = get_logger()
    logger.info("In prediction")

    # Tensorflow info logs
    logger.debug("TensorFlow version: %s", tf.__version__)
    device_name = tf.test.gpu_device_name()
    if device_name != "/device:GPU:0":
        logger.debug("GPU device not found - On for CPU time!")
    else:
        logger.debug("Found GPU at %s", device_name)

    # Load features (this may take a few mins)
    features = read_csv("data/mosaiks-features/mosaiks_features.csv")
    features = features.sort_values(by=["Lat", "Lon"])
    logger.debug(features.head())

    # Load label data
    label_data = read_csv("outputs/grid/qol-labelled-grid.csv")
    # Magic number of 0.001 to remove error between this and the features
    # TODO: Fix this error in original script
    label_data["longitude"] = round(label_data["longitude"], 3) - 0.001
    label_data["latitude"] = round(label_data["latitude"], 3) - 0.001
    label_data = label_data.sort_values(by=["latitude", "longitude"])
    logger.debug(label_data.head())

    # Merge features with label
    merged_labelled_features = merge(
        label_data,
        features,
        how="left",
        left_on=["latitude", "longitude"],
        right_on=["Lat", "Lon"],
    )

    logger.info("%s observations in merged dataframe", len(merged_labelled_features))

    # Remove invalid rows
    num_na = len(merged_labelled_features) - merged_labelled_features.dropna().shape[0]
    merged_labelled_features = merged_labelled_features.dropna()

    logger.info("%s rows with NAs dropped.", num_na)
    logger.info("%s observations in final dataframe.", len(merged_labelled_features))


if __name__ == "__main__":
    main()
