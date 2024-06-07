import os
import random
from datetime import datetime
from pathlib import Path
from typing import Dict

import geopandas as gpd
import pandas as pd
import pytz
import tensorflow as tf
from dvc.api import params_show
from dvclive import Live
from dvclive.keras import DVCLiveCallback
from keras.callbacks import TensorBoard, EarlyStopping, ModelCheckpoint
from keras.losses import MeanAbsoluteError, MeanAbsolutePercentageError
from keras.metrics import MeanSquaredError, RootMeanSquaredError
from keras.optimizers import Adam
from sklearn.model_selection import train_test_split, GroupKFold
from tensorflow.keras import Model
from tensorflow.keras import layers
from tensorflow.keras.applications import ResNet50V2

from keras_data_format_utils import create_generator
from utils.logger import get_logger
from utils.tensorflow_utils import log_tf_gpu

logger = get_logger()

# Get DVC params
params = params_show()


def load_train_dataset():
    """
    Loads training data
    """

    dataset = gpd.read_file("outputs/model/train-test-split.geojson")
    # Have to reset index here otherwise group split fails
    train = dataset[dataset["split"] == "train"].reset_index()
    return train


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
    # TODO: Write gradients deprecated?
    tensorboard_callback = TensorBoard(log_dir=logdir, write_grads=True)
    # use tensorboard --logdir logs in your command line to startup tensorboard with the correct logs

    early_stopping_callback = EarlyStopping(
        monitor="val_mean_absolute_percentage_error",
        min_delta=1,  # model should improve by at least 1%
        patience=10,  # amount of epochs  with improvements worse than 1% until the model stops
        verbose=2,
        mode="min",
        restore_best_weights=True,  # restore the best model with the lowest validation error
    )

    # TODO: Different model name depending on index?
    model_checkpoint_callback = ModelCheckpoint(
        model_path,
        monitor="val_mean_absolute_percentage_error",
        verbose=0,
        save_best_only=True,  # save the best model
        mode="min",
        save_freq="epoch",  # save every epoch
    )

    return [tensorboard_callback, early_stopping_callback, model_checkpoint_callback]


# TODO: Refactor into class model building
def resnet_model():
    """
    Defines the model
    """
    inputs = layers.Input(shape=(256, 256, 3))

    # Using ResNet50 architecture - freezing base model
    base_model = ResNet50V2(input_tensor=inputs, weights="imagenet", include_top=False)
    base_model.trainable = False

    # Rebuild top
    x = layers.GlobalAveragePooling2D(name="avg_pool")(base_model.output)
    x = layers.BatchNormalization()(x)
    top_dropout_rate = 0.2
    x = layers.Dropout(top_dropout_rate, name="top_dropout")(x)

    # final layer, since we are doing regression we will add only one neuron (unit)
    # TODO: Add more layers
    outputs = layers.Dense(1, name="pred")(x)

    # Compile
    base_model = Model(inputs, outputs, name="ResNet50V2")

    return base_model


# Custom r2 needed because tf 2.11 does not have this as a metric
# Additionally, have to use tf 2.11 because of cluster constraints
def r_squared(y_true, y_pred):
    """Custom metric function to calculate R-squared."""
    ss_res = tf.reduce_mean(tf.square(y_true - y_pred))
    ss_tot = tf.reduce_mean(tf.square(tf.math.subtract(y_true, tf.reduce_mean(y_true))))
    return 1 - ss_res / (ss_tot + tf.keras.backend.epsilon())


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
    fold :

    Returns
    -------
    Score
        The score of the model
    """

    # Training label
    training_label = params["train"]["label"]

    # Get data generators
    train_generator = create_generator(train, training_label, apply_augmentation_flag=1)
    validation_generator = create_generator(val, training_label)

    # Set model name
    # TODO: put this in params? Put it in class?
    model_path = "./outputs/model/final.h5"
    exp_dir = "dvclive"

    # If cross-validation, set model to current fold
    if fold >= 0:
        results_dir = Path("./outputs/model/folds")
        if not os.path.isdir(results_dir):
            os.makedirs(results_dir)
        model_path = f"{results_dir}/fold_{fold}.h5"
        exp_dir += f"/fold_{fold}"

    model = resnet_model()
    # model.summary()
    # TODO: Install missing packages pydot + graphviz
    # plot_model(model, to_file=f"outputs/misc/{model_name}.jpg", show_shapes=True)

    loss = params["train"]["loss"]
    model.compile(
        optimizer=Adam(),
        loss=loss,
        metrics=[
            MeanAbsoluteError(),
            MeanAbsolutePercentageError(),
            MeanSquaredError(),
            RootMeanSquaredError(),
            r_squared,
        ],
    )

    # Training label
    epochs = params["train"]["epochs"]

    with Live(save_dvc_exp=True, dir=exp_dir) as live:
        callbacks = get_callbacks(model_path)
        callbacks.append(DVCLiveCallback(save_dvc_exp=True, live=live))

        # Fit data to model
        # Note - don't worry about batch size because dataset is in the form of generators which already has batches
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

    # TODO: Move TEST evaluation to standalone

    return score_dictionary


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

    fold_scores = []
    for index, (train_index, val_index) in enumerate(gkf.split(df, groups=groups)):
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

    # TODO: Retrain the model, but this time with all the data
    #  - i.e., without making the train/test split.
    #  Save that model, and use it for generating predictions.
    # live.log_metric("test_loss", test_loss, plot=False)
    # live.log_metric("test_acc", test_acc, plot=False)


def data_split_simple(df: pd.DataFrame) -> None:
    """
    Accepts a Pandas DataFrame and splits it into training and validation. Returns DataFrames.

    Parameters
    ----------
    df : pd.DataFrame
        Pandas DataFrame containing all data.

    Returns
    -------
    None
    """
    logger.info("Training model with simple random split")
    random_state = params["constants"]["random_seed"]

    # Calculate dataset size for validation
    # We're applying a split on a reduced dataset - percentage is 1 - hold_out_test_size
    # Then we want the test_size here to be a percentage of the training set that is equal to the overall percentage
    # ie split_test_size * (1 - hold_out) = val_size
    val_size = params["split"]["val_size"]
    hold_out_test_size = params["split"]["test_size"]
    split_test_size = val_size / (1 - hold_out_test_size)

    train, val = train_test_split(
        df, test_size=split_test_size, random_state=random_state
    )

    logger.info("Descriptive statistics of train:")
    logger.info(f"Shape: {train.shape}")
    logger.info(train.describe())

    logger.info("Descriptive statistics of validation:")
    logger.info(f"Shape: {val.shape}")
    logger.info(val.describe())

    # Run model
    run_model(train, val)


def main() -> None:
    logger.info("In training")
    log_tf_gpu()

    # Seeding
    seed = params["constants"]["random_seed"]
    random.seed(seed)
    tf.random.set_seed(seed)

    # Load training dataset
    dataset = load_train_dataset()

    # Split training data into train and validation datasets
    # Type of dataset split
    group_by_ward = params["train"]["group_by_ward"]

    # If group by ward
    if group_by_ward:
        logger.info("Grouping by ward")
        # Splits and runs model within this
        data_split_ward_group_stratified_k_fold(dataset)
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
