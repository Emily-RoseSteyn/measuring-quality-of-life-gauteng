import os
import random
from pathlib import Path

import geopandas as gpd
import pandas as pd
import tensorflow as tf
from keras.layers import GlobalAveragePooling2D, Dropout, Dense
from keras.losses import MeanAbsoluteError, MeanAbsolutePercentageError
from keras.preprocessing.image import ImageDataGenerator
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split
from tensorflow.keras import Input, Model
from tensorflow.keras.applications import ResNet50V2

from utils.logger import get_logger
from utils.tensorflow_utils import log_tf_gpu

logger = get_logger()


def load_dataset():
    """
    Loads the data from path
    """

    labels = gpd.read_file("outputs/matched/gauteng-qol-cluster-tiles.geojson")
    labels = labels[["tile", "qol_index"]]
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

    return train, val, test


def get_mean_baseline(train: pd.DataFrame, val: pd.DataFrame) -> float:
    """
    Adapted from https://rosenfelder.ai/keras-regression-efficient-net/
    Calculates the mean MAE and MAPE baselines by taking the mean values of the training data
    as a naive prediction for the validation target feature.
    (ie if the model predicted mean values, what would the error be in that case)

    Parameters
    ----------
    train : pd.DataFrame
        Pandas DataFrame containing your training data.
    val : pd.DataFrame
        Pandas DataFrame containing your validation data.

    Returns
    -------
    float
        MAPE value.
    """
    # y_hat is the dummy predicted variable set to the mean of the qol index data in the training set
    y_hat = train["qol_index"].mean()
    val["y_hat"] = y_hat
    mae = MeanAbsoluteError()
    mae = mae(val["qol_index"], val["y_hat"]).numpy()
    mape = MeanAbsolutePercentageError()
    mape = mape(val["qol_index"], val["y_hat"]).numpy()

    logger.info(f"Mean Absolute Error baseline: {mae}")
    logger.info(f"Mean Absolute Percentage Error baseline: {mape}")

    return mape


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


# TODO: Refactor into class model building
def create_model(input_shape):
    """
    Defines the model
    """
    # Using ResNet50 architecture - freezing base model
    base_model = ResNet50V2(input_shape=input_shape, weights="imagenet", include_top=False)
    base_model.trainable = False

    # Create new model on top
    # Specify input shape
    inputs = Input(shape=input_shape)

    # New model is base model with training set to false
    x = base_model(inputs, training=False)
    # Add averaging layer to ensure fixed size vector
    x = GlobalAveragePooling2D()(x)
    # Add dropout layer to reduce overfitting
    x = Dropout(0.2)(x)

    # final layer, since we are doing regression we will add only one neuron (unit)
    outputs = Dense(1, activation="relu")(x)
    added_model = Model(inputs, outputs)

    return base_model, added_model


def main() -> None:
    logger.info("In training")
    log_tf_gpu()

    # Seeding
    seed = 42
    random.seed(seed)
    # np.random.seed(seed)  # TODO: Figure out how to fix this
    tf.random.set_seed(seed)

    # Load dataset
    dataset = load_dataset()
    # Split data into training, validation, test datasets
    train, val, test = split_data(dataset)

    # A naive benchmark to compare results to
    # Uses the mean of the training data as the predicted value for all x values and calculates error based on that
    get_mean_baseline(train, val)

    # Get data generators
    train_generator, validation_generator, test_generator = create_generators(
        df=dataset, train=train, val=val, test=test, visualize_augmentations_flag=1
    )

    # # TODO: Add data augmentation
    #
    # # Creating model
    # input_shape = (256, 256, 3)
    # base_model, model = create_model(input_shape=input_shape)
    #
    # model.compile(
    #     optimizer=Adam(),
    #     loss=MeanSquaredError(),
    # )
    #
    # # TODO: What am I doing after this??
    #
    # # checkpoint
    # filepath = "../outputs/checkpoints/weights-improvement-{epoch:02d}-{val_loss:.2f}.hdf5"
    # checkpoint = ModelCheckpoint(filepath, monitor="val_loss", verbose=1, save_best_only=True, mode="max")
    # callbacks_list = [checkpoint]
    #
    # # Top layer fit
    # epochs = 20
    # # print("Fitting the top layer of the model")
    # # TODO: What does this mean again?
    # model.fit(train, epochs=epochs, validation_data=validation, batch_size=10, callbacks=callbacks_list)
    #
    # # Unfreeze the base_model. Note that it keeps running in inference mode
    # # since we passed `training=False` when calling it. This means that
    # # the batchnorm layers will not update their batch statistics.
    # # This prevents the batchnorm layers from undoing all the training
    # # we've done so far.
    # base_model.trainable = True
    # model.summary(show_trainable=True)
    #
    # model.compile(
    #     optimizer=Adam(1e-5),  # Low learning rate
    #     loss=MeanSquaredError(),
    # )
    #
    # epochs = 100
    # # print("Fitting the end-to-end model")
    # model.fit(train, epochs=epochs, validation_data=validation)
    # # with Live() as live:
    # #     model.fit(
    # #         train,
    # #         validation_data=validation,
    # #         callbacks=[
    # #             DVCLiveCallback(live=live)
    # #         ]
    # #     )
    # #     model.save("model")
    # #     live.log_artifact("model", type="model")


if __name__ == "__main__":
    main()
