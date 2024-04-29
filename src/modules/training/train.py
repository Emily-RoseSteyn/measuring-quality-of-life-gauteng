import random

import geopandas as gpd
import tensorflow as tf
from keras.callbacks import ModelCheckpoint
from keras.layers import GlobalAveragePooling2D, Dropout, Dense
from keras_preprocessing.image import ImageDataGenerator
from tensorflow.keras import Input, Model
from tensorflow.keras.applications import ResNet50V2
from tensorflow.keras.losses import MeanSquaredError
from tensorflow.keras.optimizers import Adam

from utils.logger import get_logger

logger = get_logger()


# TODO: Refactor to return train, validation AND TEST
# Perhaps also find a better way to do this so that you can return tuple instead of calling function separately
def load_dataset(subset):
    """
    Loads the subset (training/validation) of the data from path
    """

    labels = gpd.read_file("outputs/matched/gauteng-qol-cluster-tiles.geojson")
    labels = labels[["tile", "qol_index"]]
    data = ImageDataGenerator(validation_split=0.2, rescale=1 / 255)
    data_flow = data.flow_from_dataframe(
        dataframe=labels,
        directory="outputs/tiles",
        x_col="tile",
        y_col="qol_index",
        target_size=(256, 256),
        batch_size=32,
        class_mode="raw",
        subset=subset,
        seed=42)

    return data_flow


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

    # Load training and validation datasets (actually this is validation data set)
    train = load_dataset("training")
    validation = load_dataset("validation")

    # Creating model
    input_shape = (256, 256, 3)
    base_model, model = create_model(input_shape=input_shape)

    model.compile(
        optimizer=Adam(),
        loss=MeanSquaredError(),
    )

    # TODO: What am I doing after this??

    # checkpoint
    filepath = "../outputs/checkpoints/weights-improvement-{epoch:02d}-{val_loss:.2f}.hdf5"
    checkpoint = ModelCheckpoint(filepath, monitor="val_loss", verbose=1, save_best_only=True, mode="max")
    callbacks_list = [checkpoint]

    # Top layer fit
    epochs = 20
    # print("Fitting the top layer of the model")
    # TODO: What does this mean again?
    model.fit(train, epochs=epochs, validation_data=validation, batch_size=10, callbacks=callbacks_list)

    # Unfreeze the base_model. Note that it keeps running in inference mode
    # since we passed `training=False` when calling it. This means that
    # the batchnorm layers will not update their batch statistics.
    # This prevents the batchnorm layers from undoing all the training
    # we've done so far.
    base_model.trainable = True
    model.summary(show_trainable=True)

    model.compile(
        optimizer=Adam(1e-5),  # Low learning rate
        loss=MeanSquaredError(),
    )

    epochs = 100
    # print("Fitting the end-to-end model")
    model.fit(train, epochs=epochs, validation_data=validation)
    # with Live() as live:
    #     model.fit(
    #         train,
    #         validation_data=validation,
    #         callbacks=[
    #             DVCLiveCallback(live=live)
    #         ]
    #     )
    #     model.save("model")
    #     live.log_artifact("model", type="model")


if __name__ == "__main__":
    main()
