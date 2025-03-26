import os
import random
from datetime import datetime
from pathlib import Path
from typing import Dict

import pandas as pd
import pytz
import tensorflow as tf
from dvc.api import params_show
from keras.src.callbacks import EarlyStopping, ModelCheckpoint, TensorBoard
from keras.src.losses import (
    MeanAbsoluteError,
    MeanAbsolutePercentageError,
    MeanSquaredError,
)
from keras.src.metrics import R2Score, RootMeanSquaredError
from keras.src.optimizers import Adam
from models.model_factory import ModelFactory
from sklearn.model_selection import GroupKFold
from utils.keras_data_format import create_generator
from utils.load_processed_data import load_dataset
from utils.logger import get_logger
from utils.tensorflow_utils import log_tf_gpu
from utils.test_data_split import (
    test_data_split_simple_random,
    test_data_split_ward_group_shuffle_split,
)

from dvclive import Live
from dvclive.keras import DVCLiveCallback

logger = get_logger()

# Get DVC params
params = params_show()


def get_callbacks(model_path: str) -> list:
    """
    Adapted from https://rosenfelder.ai/keras-regression-efficient-net/
    Accepts the model name as a string and returns multiple callbacks for training the keras model.

    Parameters
    ----------
    model_path : str
        The path of where to save the model as a string.

    Returns
    -------
    List[Union[TensorBoard, EarlyStopping, ModelCheckpoint]]
        A list of multiple keras callbacks.
    """
    logdir = (
            "logs/scalars/"
            + model_path
            + "_"
            + datetime.now(tz=pytz.utc).strftime("%Y%m%d-%H%M%S")
    )  # create a folder for each model.
    tensorboard_callback = TensorBoard(log_dir=logdir,
                             histogram_freq=1,
                             write_graph=True,
                             write_images=True,
                             update_freq="epoch",
                             profile_batch=2,
                             embeddings_freq=1)
    # use tensorboard --logdir logs in your command line to startup tensorboard with the correct logs

    # TODO: Something wrong with early stopping?
    early_stopping_callback = EarlyStopping(
        min_delta=1,  # model should improve by at least 1%
        patience=10,  # amount of epochs  with improvements worse than 1% until the model stops
        verbose=2,
        mode="min",
        restore_best_weights=True,  # restore the best model with the lowest validation error
    )

    # TODO: Different model name depending on index?
    model_checkpoint_callback = ModelCheckpoint(
        model_path,
        verbose=0,
        save_best_only=True,  # save the best model
        mode="min",
        save_freq="epoch",  # save every epoch
    )

    return [tensorboard_callback, early_stopping_callback, model_checkpoint_callback]


# TODO: Different models? Fine-tuning? Generalisation (see blog)
def run_model(
        train: pd.DataFrame, val: pd.DataFrame, fold: int = -1
) -> Dict[str, float]:  # noqa: FA100
    """
    This function runs a keras model with the Adam optimizer and multiple callbacks.
    The model is evaluated within training through the validation generator.

    Parameters
    ----------
    train : Pandas Dataframe
        keras Pandas Dataframe for the training data.
    val : Pandas Dataframe
        keras Pandas Dataframe for the validation data.
    fold : int
        The fold that is currently running
        Set to -1 if no cross-fold validation

    Returns
    -------
    Score
        The score of the model
    """

    # Training label
    training_label = params["train"]["label"]

    # Get data generators
    apply_augmentations = params["train"]["augment_training_data"]
    train_generator = create_generator(
        train, training_label, apply_augmentation_flag=apply_augmentations
    )
    validation_generator = create_generator(val, training_label)

    # Set model name
    # TODO: put this in params? Put it in class?
    output_dir = "./outputs/model"
    model_path = f"{output_dir}/final.h5"
    exp_dir = "dvclive"

    # If cross-validation, set model to current fold
    if fold >= 0:
        fold_dir = Path(f"{output_dir}/folds")
        if not os.path.isdir(fold_dir):
            os.makedirs(fold_dir)
        model_path = f"{fold_dir}/fold_{fold}.h5"
        exp_dir += f"/fold_{fold}"

    # Dynamically create model class from model name
    model_name = params["train"]["model_name"]
    model_class = ModelFactory.get(model_name)
    model_class.save_model_summary(output_dir)

    # Get actual keras model
    model = model_class.keras_model

    # TODO: Consider moving into base model/child model classes?
    loss = params["train"]["loss"]
    learning_rate = params["train"]["learning_rate"]
    model.compile(
        optimizer=Adam(learning_rate=learning_rate),
        loss=loss,
        metrics=[
            MeanAbsoluteError(),
            MeanAbsolutePercentageError(),
            MeanSquaredError(),
            RootMeanSquaredError(),
            R2Score()
        ],
    )

    # Training label
    epochs = params["train"]["epochs"]

    with Live(save_dvc_exp=True, dir=exp_dir) as live:
        callbacks = get_callbacks(model_path)
        callbacks.append(DVCLiveCallback(save_dvc_exp=True, live=live))

        # Fit data to model
        # Note - don't worry about batch size because dataset is in the form of generators
        # which already has batches
        model.fit(
            train_generator,
            epochs=epochs,
            validation_data=validation_generator,
            callbacks=callbacks,
        )

        live.log_artifact(model_path, type="model")

    # Generate generalization metrics
    score = model.evaluate(validation_generator, callbacks=callbacks)

    score_dictionary = dict(zip(model.metrics_names, score))
    logger.info("Validation scores")
    logger.info(score_dictionary)

    return score_dictionary


def save_updated_splits(selected_validation):
    dataset = load_dataset("all")
    dataset.loc[selected_validation, "split"] = "validation"
    dataset.to_file("outputs/model/train-validation-test-split.geojson", driver="GeoJSON")


def data_split_ward_group_stratified_k_fold(df: pd.DataFrame) -> None:
    """
    Accepts a Pandas DataFrame and splits it using the group stratified k fold method

    Parameters
    ----------
    df : pd.DataFrame
        Pandas DataFrame containing all data.

    Returns
    -------
    None
    """

    folds = params["split"]["folds"]
    # TODO: Look at continuous target stratification
    #  https://neptune.ai/blog/cross-validation-mistakes#h-3-choosing-cross-validation-technique-for-a-regression-problem
    gkf = GroupKFold(n_splits=folds)
    groups = df["ward_code"]
    split = list(gkf.split(df, groups=groups))

    fold_scores = []
    for index, (train_index, val_index) in enumerate(split):
        fold = index
        logger.info(f"Training for fold: {fold}")

        # Access train and validation grouped data
        train_groups = groups[train_index]
        val_groups = groups[val_index]

        train = df.loc[train_groups.index]
        val = df.loc[val_groups.index]

        # Run model
        score = run_model(train, val, fold)
        fold_scores.append(score)

    logger.info("----------------------------------------------------------")
    logger.info("Score per fold")
    fold_scores_df = pd.DataFrame(fold_scores)
    logger.info(fold_scores_df)

    logger.info("----------------------------------------------------------")
    logger.info("Average scores for all folds:")
    logger.info("----------------------------------------------------------")
    logger.info(fold_scores_df.mean(axis=0))

    logger.info("----------------------------------------------------------")
    logger.info("Standard deviation scores for all folds:")
    logger.info("----------------------------------------------------------")
    logger.info(fold_scores_df.std(axis=0))

    logger.info("----------------------------------------------------------")
    logger.info("Best fold:")
    logger.info("----------------------------------------------------------")
    loss_param = params["train"]["loss"]
    loss = fold_scores_df[loss_param]
    min_loss_fold = loss.idxmin()
    min_loss = loss.min()
    logger.info(f"Fold {min_loss_fold} with a loss of {min_loss}")

    # TODO: Load the best performing model instance

    result_dir = "./outputs/model"
    fold_model = Path(f"{result_dir}/folds/fold_{min_loss_fold}.h5")
    final_model = Path(f"{result_dir}/final.h5")
    os.replace(fold_model, final_model)

    # Save the train, validation split
    selected_split = next(fold for idx, fold in enumerate(split) if idx == min_loss_fold)
    selected_validation = selected_split[1]
    save_updated_splits(selected_validation)

    # TODO: Retrain the model, but this time with all the data
    #  - i.e., without making the train/test split.
    #  Save that model, and use it for generating predictions.
    # live.log_metric("test_loss", test_loss, plot=False)
    # live.log_metric("test_acc", test_acc, plot=False)


def data_split_ward_grouped(df: pd.DataFrame) -> None:
    """
    Accepts a Pandas DataFrame and splits it into training and validation where these are grouped by ward.

    Parameters
    ----------
    df : pd.DataFrame
        Pandas DataFrame containing all data.

    Returns
    -------
    None
    """
    logger.info("Training model with grouped ward split")

    # Calculate dataset size for validation
    # We're applying a split on a reduced dataset - percentage is 1 - hold_out_test_size
    # Then we want the test_size here to be a percentage of the training set that is equal to the overall percentage
    # ie split_test_size * (1 - hold_out) = val_size
    val_size = params["split"]["val_size"]
    hold_out_test_size = params["split"]["test_size"]
    split_test_size = val_size / (1 - hold_out_test_size)

    train, val = test_data_split_ward_group_shuffle_split(df, test_size=split_test_size)

    log_train_val_statistics(train, val)

    # Run model
    run_model(train, val)

    # Save updated train split
    save_updated_splits(val.index)


def data_split_simple(df: pd.DataFrame) -> None:
    """
    Accepts a Pandas DataFrame and splits it into training and validation.

    Parameters
    ----------
    df : pd.DataFrame
        Pandas DataFrame containing all data.

    Returns
    -------
    None
    """
    logger.info("Training model with simple random split")

    # Calculate dataset size for validation
    # We're applying a split on a reduced dataset - percentage is 1 - hold_out_test_size
    # Then we want the test_size here to be a percentage of the training set that is equal to the overall percentage
    # ie split_test_size * (1 - hold_out) = val_size
    val_size = params["split"]["val_size"]
    hold_out_test_size = params["split"]["test_size"]
    split_test_size = val_size / (1 - hold_out_test_size)

    train, val = test_data_split_simple_random(df, split_test_size)

    log_train_val_statistics(train, val)

    # Run model
    run_model(train, val)

    # Save updated train split
    save_updated_splits(val.index)


def log_train_val_statistics(train, val):
    logger.info("Descriptive statistics of train:")
    logger.info(f"Shape: {train.shape}")
    logger.info(train.describe())
    logger.info("Descriptive statistics of validation:")
    logger.info(f"Shape: {val.shape}")
    logger.info(val.describe())


def main() -> None:
    logger.info("In training")
    log_tf_gpu()

    # Seeding
    seed = params["constants"]["random_seed"]
    random.seed(seed)
    tf.random.set_seed(seed)

    # Load training dataset
    dataset = load_dataset("train")

    # Split training data into train and validation datasets
    # Type of dataset split
    group_by_ward = params["split"]["group_by_ward"]
    cross_val = params["train"]["cross_val"]

    # If group by ward and cross validation
    if group_by_ward and cross_val:
        logger.info("Grouping by ward and running cross-validation")
        # Splits and runs model within this
        data_split_ward_group_stratified_k_fold(dataset)
    # If group by ward
    elif group_by_ward:
        logger.info("Grouping by ward without cross-validation")
        data_split_ward_grouped(dataset)

    # If cross validation
    elif cross_val:
        logger.error("Not implemented")
        raise NotImplementedError(
            "Cross-validation without grouping wards is not implemented"
        )
    # Else assume simple random
    else:
        logger.info("Random splitting")
        data_split_simple(dataset)


if __name__ == "__main__":
    main()

# TODO: Cite possibly at least in repo?
# https://rosenfelder.ai/keras-regression-efficient-net/
# @misc{rosenfelderaikeras2020,
#   author = {Rosenfelder, Markus},
#   title = {Transfer Learning with EfficientNet for Image Regression in Keras - Using Custom Data in Keras},
#   year = {2020},
#   publisher = {rosenfelder.ai},
#   journal = {rosenfelder.ai},
#   howpublished = {\url{https://rosenfelder.ai/keras-regression-efficient-net/}},
# }
