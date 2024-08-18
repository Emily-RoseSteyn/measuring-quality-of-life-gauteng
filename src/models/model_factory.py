from models.base_model import BaseModel
from models.model_types import ModelType
from models.resnet_model import ResnetModel

ModelMap = {
    ModelType.Resnet50V2: ResnetModel,
}


class ModelFactory:

    @staticmethod
    def get(model_type: ModelType) -> BaseModel:
        creator = ModelMap.get(model_type)
        if not creator:
            raise ValueError(f"Unknown model type: {model_type}")
        return creator()
