import os

from plot import plot_actual_vs_predicted
from predict import make_predictions
from utils.env_variables import RESULTS_DIR
from utils.load_processed_data import load_dataset
from utils.logger import get_logger


def predict_and_plot(split: str):
    # Make results folder
    os.makedirs(RESULTS_DIR, exist_ok=True)

    # Load full dataset
    # TODO: Modify load to load everything rather and then can group by train/test?
    # TODO: Should save folds??
    data_split = load_dataset(split)

    # Make predictions
    data_predictions = make_predictions(data_split)

    # Plot predictions
    plot_actual_vs_predicted(data_predictions, split)

    # Map activations

    # Map some examples

    # TODO: Other plots


def main():
    logger = get_logger()
    logger.info("Predicting and plotting things")
    predict_and_plot("train")
    predict_and_plot("test")


if __name__ == "__main__":
    main()
