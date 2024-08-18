from dvc.api import params_show
from keras import layers, Model
from keras.applications import ResNet50V2
from keras.layers import GlobalAveragePooling2D, Dense, BatchNormalization, Dropout

from models.base_model import BaseModel
from models.model_types import ModelType


class ResnetModel(BaseModel):
    @property
    def name(self) -> ModelType:
        return ModelType.Resnet50V2

    @property
    def keras_model(self) -> Model:
        params = params_show()["model"][self.name]
        inputs = layers.Input(shape=(256, 256, 3))

        # Using ResNet50 architecture - freezing base model
        base_model = ResNet50V2(
            input_tensor=inputs, weights="imagenet", include_top=False
        )
        base_model.trainable = False

        # Get outputs of base model
        x = base_model.output

        # Add global average pooling layer
        x = GlobalAveragePooling2D(name="avg_pool")(x)

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
