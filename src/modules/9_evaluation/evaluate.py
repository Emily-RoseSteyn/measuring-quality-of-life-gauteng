import keras
from dvc.api import params_show
from dvclive import Live
from tensorflow.keras.models import load_model

from utils.keras_data_format import create_generator
from utils.load_processed_data import load_dataset
from utils.logger import get_logger
from utils.r2_score import r_squared

logger = get_logger()

# Get DVC params
params = params_show()


def evaluate(model: keras.Model, dataset, split, live):
    """
    Dump all evaluation metrics and plots for given datasets.

    Args:
        model: Trained model
        dataset: The input dataset to evaluate
        split (str): Dataset name.
        live (dvclive.Live): Dvclive instance.
    """

    # Training label
    training_label = params["train"]["label"]

    # Get data generators
    generator = create_generator(dataset, training_label)

    # Generate generalization metrics
    score = model.evaluate(generator)

    score_dictionary = dict(zip(model.metrics_names, score))
    logger.info(f"{split} scores")
    logger.info(score_dictionary)

    # Log to DVC
    for key, value in score_dictionary.items():
        live.log_metric(f"{split}_{key}", value, plot=False)


def main() -> None:
    logger.info("In evaluate")

    eval_path = "outputs/eval"
    model_file = "outputs/model/final.h5"

    # Load model
    model = load_model(model_file, custom_objects={r_squared})

    # Load datasets
    # train = load_dataset("train")
    test = load_dataset("test")

    # Evaluate train and test datasets.
    with Live(dir=eval_path) as live:
        # evaluate(model, train, "train", live)
        evaluate(model, test, "test", live)


if __name__ == "__main__":
    main()
