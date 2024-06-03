import json
import os
import random
from datetime import datetime
from pathlib import Path

import geopandas as gpd
import pandas as pd
import pytz
import tensorflow as tf
from dvc.api import params_show
from dvclive import Live
from dvclive.keras import DVCLiveCallback
from keras.callbacks import History, TensorBoard, EarlyStopping, ModelCheckpoint
from keras.losses import MeanAbsoluteError, MeanAbsolutePercentageError
from keras.metrics import MeanSquaredError, RootMeanSquaredError
from keras.optimizers import Adam
from keras.preprocessing.image import ImageDataGenerator, Iterator
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split
from tensorflow.keras import Model
from tensorflow.keras import layers
from tensorflow.keras.applications import ResNet50V2

from utils.logger import get_logger
from utils.tensorflow_utils import log_tf_gpu

logger = get_logger()

# Get DVC params
params = params_show()


def load_train_dataset():
    """
    Loads training data
    """

    training_label = params["train"]["label"]
    dataset = gpd.read_file("outputs/model/train-test-split.geojson")
    train = dataset[dataset["split"] == "train"]
    labels = train[["tile", training_label]]
    return labels


def split_data(df: pd.DataFrame) -> tuple:
    """
    Accepts a Pandas DataFrame and splits it into training, validation, and test data. Returns DataFrames.

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
    # TODO: Distribution of these datasets needs to be fair?
    #  See e2e notebooks with stratified shuffle split
    train, val = train_test_split(df, test_size=0.2, random_state=1)  # split the data with a validation size o 20%
    train, test = train_test_split(
        train, test_size=0.125, random_state=1
    )  # split the data with an overall  test size of 10%

    logger.info("Descriptive statistics of train:")
    logger.info(f"Shape: {train.shape}")
    logger.info(train.describe())

    logger.info("Descriptive statistics of validation:")
    logger.info(f"Shape: {val.shape}")
    logger.info(val.describe())

    logger.info("Descriptive statistics of test:")
    logger.info(f"Shape: {test.shape}")
    logger.info(test.describe())

    df.loc[train.index, "split"] = "train"
    df.loc[val.index, "split"] = "val"
    df.loc[test.index, "split"] = "test"
    df.to_csv("outputs/model/data-split.csv")

    return train, val, test


# TODO: Clean up and move elsewhere
def visualize_augmentations(data_generator: ImageDataGenerator, df: pd.DataFrame):
    """
    Adapted from https://rosenfelder.ai/keras-regression-efficient-net/
    Visualizes the keras augmentations with matplotlib in 3x3 grid. This function is part of create_generators() and
    can be accessed from there.

    Parameters
    ----------
    data_generator : Iterator
        The keras data generator of your training data.
    df : pd.DataFrame
        The Pandas DataFrame containing your training data.
    """
    image_dir = os.path.abspath(Path("./outputs/misc"))

    if not os.path.isdir(image_dir):
        os.makedirs(image_dir)

    # super hacky way of creating a small dataframe with one image
    series = df.iloc[2]

    df_augmentation_visualization = pd.concat([series, series], axis=1).transpose()

    iterator_visualizations = data_generator.flow_from_dataframe(
        directory="outputs/tiles",
        dataframe=df_augmentation_visualization,
        x_col="tile",
        y_col="qol_index",
        class_mode="raw",
        target_size=(256, 256),
        batch_size=1,  # use only one image for visualization
    )

    for i in range(9):
        plt.subplot(3, 3, i + 1)  # create a 3x3 grid
        batch = next(iterator_visualizations)  # get the next image of the generator (always the same image)
        img = batch[0]
        img = img[0, :, :, :]  # remove one dimension for plotting without issues
        plt.imshow(img)
    plt.savefig("outputs/misc/augmentations.png")


def create_generators(
        df: pd.DataFrame, train: pd.DataFrame, val: pd.DataFrame, test: pd.DataFrame,
        visualize_augmentations_flag: int = 0
) -> tuple:
    """
    Adapted from https://rosenfelder.ai/keras-regression-efficient-net/
    Accepts four Pandas DataFrames: all data, the training, validation and test DataFrames. Creates and returns
    keras ImageDataGenerators.
    The augmentations of the ImageDataGenerators can also be visualised in this function.

    Parameters
    ----------
    df : pd.DataFrame
        Pandas DataFrame containing all data.
    train : pd.DataFrame
        Pandas DataFrame containing training data.
    val : pd.DataFrame
        Pandas DataFrame containing validation data.
    test : pd.DataFrame
        Pandas DataFrame containing testing data.
    visualize_augmentations_flag: int
        Flag to visualise augmentations.

    Returns
    -------
    tuple[Iterator, Iterator, Iterator]
        keras ImageDataGenerators used for training, validating and testing of your models.
    """
    # Create training ImageDataGenerator with image augmentations
    # TODO: Check image augmentation
    train_generator = ImageDataGenerator(
        rescale=1.0 / 255,
        rotation_range=5,
        width_shift_range=0.1,
        height_shift_range=0.1,
        brightness_range=(0.75, 1),
        shear_range=0.1,
        zoom_range=[0.75, 1],
        horizontal_flip=True,
        vertical_flip=True
    )

    # Visualize image augmentations if flag set before actually getting dataframe
    if visualize_augmentations_flag == 1:
        visualize_augmentations(train_generator, df)

    # Create dataframe iterator
    train_generator = train_generator.flow_from_dataframe(
        directory="outputs/tiles",
        dataframe=train,
        x_col="tile",  # Image location
        y_col="qol_index",  # Target feature
        class_mode="raw",  # Use "raw" for regressions TODO: Understand why?
        target_size=(256, 256),
        # TODO: increase or decrease to fit GPU
        batch_size=32,
    )

    # Create validation and test ImageDataGenerators without image augmentations
    # Except for rescaling, no augmentations are needed for validation and testing generators
    validation_generator = ImageDataGenerator(
        rescale=1.0 / 255
    )
    validation_generator = validation_generator.flow_from_dataframe(
        directory="outputs/tiles",
        dataframe=val,
        x_col="tile",  # Image location
        y_col="qol_index",  # Target feature
        class_mode="raw",
        target_size=(256, 256),
        batch_size=32,
    )

    test_generator = ImageDataGenerator(rescale=1.0 / 255)
    test_generator = test_generator.flow_from_dataframe(
        directory="outputs/tiles",
        dataframe=test,
        x_col="tile",  # Image location
        y_col="qol_index",  # Target feature
        class_mode="raw",
        target_size=(256, 256),
        batch_size=32,
    )
    return train_generator, validation_generator, test_generator


def get_callbacks(model_name: str) -> list:
    """
    Adapted from https://rosenfelder.ai/keras-regression-efficient-net/
    Accepts the model name as a string and returns multiple callbacks for training the keras model.

    Parameters
    ----------
    model_name : str
        The name of the model as a string.

    Returns
    -------
    List[Union[TensorBoard, EarlyStopping, ModelCheckpoint]]
        A list of multiple keras callbacks.
    """
    logdir = (
            "logs/scalars/" + model_name + "_" + datetime.now(tz=pytz.utc).strftime("%Y%m%d-%H%M%S")
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

    model_checkpoint_callback = ModelCheckpoint(
        "./outputs/model/" + model_name + ".h5",
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
    inputs = layers.Input(
        shape=(256, 256, 3)
    )

    # Using ResNet50 architecture - freezing base model
    base_model = ResNet50V2(input_tensor=inputs, weights="imagenet", include_top=False)
    base_model.trainable = False

    # Rebuild top
    x = layers.GlobalAveragePooling2D(name="avg_pool")(base_model.output)
    x = layers.BatchNormalization()(x)
    top_dropout_rate = 0.2
    x = layers.Dropout(top_dropout_rate, name="top_dropout")(x)

    # final layer, since we are doing regression we will add only one neuron (unit)
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
        train_generator: Iterator,
        validation_generator: Iterator,
        test_generator: Iterator,
) -> History:
    """
    This function runs a keras model with the Adam optimizer and multiple callbacks. The model is evaluated within
    training through the validation generator and one final time on the test generator at the end of fitting.

    Parameters
    ----------
    train_generator : Iterator
        keras ImageDataGenerators for the training data.
    validation_generator : Iterator
        keras ImageDataGenerators for the validation data.
    test_generator : Iterator
        keras ImageDataGenerators for the test data.

    Returns
    -------
    History
        The history of the keras model as a History object. To access it as a Dict, use history.history. For an example
        see plot_results().
    """

    model_name = "resnet"

    callbacks = get_callbacks(model_name)
    model = resnet_model()
    # model.summary()
    # TODO: Install missing packages pydot + graphviz
    # plot_model(model, to_file=f"outputs/misc/{model_name}.jpg", show_shapes=True)

    # TODO: Different optimizers?
    model.compile(
        optimizer=Adam(), loss="mean_absolute_error",
        metrics=[MeanAbsoluteError(), MeanAbsolutePercentageError(), MeanSquaredError(), RootMeanSquaredError(),
                 r_squared]
    )
    with Live(save_dvc_exp=True) as live:
        callbacks.append(DVCLiveCallback(save_dvc_exp=True, live=live))
        history = model.fit(
            train_generator,
            epochs=100,
            validation_data=validation_generator,
            callbacks=callbacks,
        )
        live.log_artifact(model_name, type="model")

    # Dump the dictionary containing each metric and the loss for each epoch
    history_save_path = f"./outputs/model/{model_name}-history.json"
    with open(history_save_path, "w") as history_file:
        json.dump(history.history, history_file)

    score = model.evaluate(test_generator,
                           callbacks=callbacks)
    logger.info("Test scores")
    logger.info(score)
    return history


def main() -> None:
    logger.info("In training")
    log_tf_gpu()

    # Seeding
    seed = params["constants"]["random_seed"]
    random.seed(seed)
    tf.random.set_seed(seed)

    # Load training dataset
    dataset = load_train_dataset()

    # Split data into training, validation, test datasets
    train, val, test = split_data(dataset)

    # Get data generators
    train_generator, validation_generator, test_generator = create_generators(
        df=dataset, train=train, val=val, test=test, visualize_augmentations_flag=1
    )

    # Run model
    run_model(
        train_generator=train_generator,
        validation_generator=validation_generator,
        test_generator=test_generator,
    )


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
