from dvc.api import params_show
from keras import layers, Model
from keras.applications import VGG16
from keras.layers import Dense, BatchNormalization, Dropout

from models.base_model import BaseModel
from models.model_types import ModelType
from utils.env_variables import TILE_SIZE_WITH_CHANNELS


class VGG16Model(BaseModel):
    @property
    def name(self) -> ModelType:
        return ModelType.VGG16

    @property
    def keras_model(self) -> Model:
        params = params_show()["model"][self.name]
        inputs = layers.Input(shape=TILE_SIZE_WITH_CHANNELS)

        # Using VGG16 architecture - freezing base model
        # Potentially don't initialise to imagenet - see paper; Glorot Normal random initialization
        base_keras_model = VGG16(
            input_tensor=inputs, weights="imagenet", include_top=False, pooling="avg"
        )
        base_keras_model.trainable = False

        # Get outputs of base model
        x = base_keras_model.output

        # Add a fully connected layer
        fc_units = params["fc_units"]
        x = Dense(fc_units, activation="relu", name="fc")(x)

        # Add batch normalization
        x = BatchNormalization(name="batch_norm")(x)

        # Add dropout
        dropout_rate = params["dropout_rate"]
        x = Dropout(dropout_rate, name="top_dropout")(x)

        # Add the final output layer for regression
        predictions = Dense(1, activation="linear", name="pred")(x)

        # Create the new model
        return Model(inputs, outputs=predictions, name=self.name)
