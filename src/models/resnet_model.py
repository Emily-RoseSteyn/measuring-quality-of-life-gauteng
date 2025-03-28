from dvc.api import params_show
from keras import Model, layers
from keras.src.applications.resnet_v2 import ResNet50V2
from keras.src.layers import Dense
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
        # NB! Adding pooling here so don't need output pooling
        base_keras_model = ResNet50V2(
            input_tensor=inputs, weights="imagenet", include_top=False, pooling="avg"
        )
        base_keras_model.trainable = params["train_base"]

        # Get outputs of base model
        x = base_keras_model.output

        # Add a fully connected layer with an output unit
        activation = params["activation"]
        predictions = Dense(1, activation=activation, name="pred")(x)

        # Create the new model
        return Model(inputs, outputs=predictions, name=self.name)
