from tensorflow.keras.models import load_model

from utils import r2_score_wrapper
from utils.load_processed_data import load_dataset
from utils.logger import get_logger


def main():
    logger = get_logger()
    logger.info("Predicting and plotting things")

    # Results directory where everything is
    model_file = "outputs/model/final.h5"

    # Load model
    model = load_model(model_file, custom_objects={"r_squared": r2_score_wrapper})

    # Load full dataset
    # TODO: Modify load to load everything rather and then can group by train/test?
    # TODO: Should save folds??
    train = load_dataset("train")
    test = load_dataset("test")

    return train, test, model

    # Make predictions

    # Plot predictions

    # Map activations

    # Map some examples

    # TODO: Other plots


if __name__ == "__main__":
    main()
