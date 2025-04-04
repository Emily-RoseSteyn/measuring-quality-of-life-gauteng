import pickle

import keras
from dvc.api import params_show
from keras.src.saving import load_model
from utils.keras_data_format import create_generator
from utils.load_processed_data import load_dataset
from utils.logger import get_logger

from dvclive import Live

logger = get_logger()

# Get DVC params
params = params_show()


def evaluate(model: keras.Model, split, eval_dir: str, live: int = 0):
    """
    Dump all evaluation metrics and plots for given datasets.

    Args:
        model: Trained model
        split (str): Dataset name.
        eval_dir (str): the directory to save results to
        live (int): Whether to start a dvclive instance.
    """
    dataset = load_dataset(split, post_training=1)
    # Training label
    training_label = params["train"]["label"]

    # Get data generators
    generator = create_generator(dataset, training_label)

    # Generate generalization metrics
    score = model.evaluate(generator)

    score_dictionary = dict(zip(model.metrics_names, score))
    logger.info(f"{split} scores")
    logger.info(score_dictionary)

    # Save scores
    with open(f"{eval_dir}/{split}_scores.pkl", "wb") as f:
        pickle.dump(score_dictionary, f)

    # Log to DVC
    if live:
        with Live(dir=eval_dir) as live_instance:
            for key, value in score_dictionary.items():
                live_instance.log_metric(f"{split}_{key}", value, plot=False)


def main() -> None:
    logger.info("In evaluate")

    eval_path = "outputs/eval"
    model_file = "outputs/model/final.keras"

    # Load model
    model = load_model(model_file)

    # Evaluate all datasets
    evaluate(model, "train", eval_path)
    evaluate(model, "validation", eval_path)
    evaluate(model, "test", eval_path, live=1)
    # TODO: Is it evaluation on each fold?? Or in simple split, only on val?
    # For now just doing validation/train (selected from best fold in k-fold


if __name__ == "__main__":
    main()
