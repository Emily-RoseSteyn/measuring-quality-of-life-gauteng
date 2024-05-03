import random

import geopandas as gpd
import pandas as pd
import tensorflow as tf
from keras.layers import GlobalAveragePooling2D, Dropout, Dense
from keras.losses import MeanAbsoluteError, MeanAbsolutePercentageError
from sklearn.model_selection import train_test_split
from tensorflow.keras import Input, Model
from tensorflow.keras.applications import ResNet50V2

from utils.logger import get_logger

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
