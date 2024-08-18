from abc import ABC, abstractmethod
from contextlib import redirect_stdout

from keras import Model

from models.model_types import ModelType


class BaseModel(ABC):

    @property
    @abstractmethod
    def keras_model(self) -> Model:
        pass

    @property
    @abstractmethod
    def name(self) -> ModelType:
        pass

    def save_model_summary(self, output_dir):
        with open(f"{output_dir}/model_summary.txt", "w") as f, redirect_stdout(f):
            self.keras_model.summary()

        # TODO: Plot model - install missing packages pydot + graphviz
        # plot_model(model, to_file=f"outputs/misc/{model_name}.jpg", show_shapes=True)
