from keras import layers, Model
from keras.applications import ResNet50V2


class ResnetModel:
    def __init__(self):
        """
        Defines the model
        """
        self.name = "ResNet50V2"
        self.inputs = layers.Input(shape=(256, 256, 3))

        # Using ResNet50 architecture - freezing base model
        model = ResNet50V2(
            input_tensor=self.inputs, weights="imagenet", include_top=False
        )
        model.trainable = False

        # Rebuild top
        x = layers.GlobalAveragePooling2D(name="avg_pool")(model.output)
        x = layers.BatchNormalization()(x)
        top_dropout_rate = 0.2
        x = layers.Dropout(top_dropout_rate, name="top_dropout")(x)

        # final layer, since we are doing regression we will add only one neuron (unit)
        # TODO: Add more layers
        outputs = layers.Dense(1, name="pred")(x)

        # Compile
        model = Model(self.inputs, outputs, name=self.name)

        self.model = model
