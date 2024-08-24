from dvc.api import params_show
from keras import layers, Model
from keras.applications import ResNet50V2
from keras.layers import Dense, BatchNormalization, Dropout

from models.base_model import BaseModel
from models.model_types import ModelType
from utils.env_variables import TILE_SIZE_WITH_CHANNELS


class ResnetModel(BaseModel):
    @property
    def name(self) -> ModelType:
        return ModelType.Resnet50V2

    @property
    def keras_model(self) -> Model:
        params = params_show()["model"][self.name]
        inputs = layers.Input(shape=TILE_SIZE_WITH_CHANNELS)

        # Using ResNet50 architecture - freezing base model
        base_keras_model = ResNet50V2(
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
