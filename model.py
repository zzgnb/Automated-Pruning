from importlib import import_module
import torch
from loguru import logger
from config import DfParams, config

# TODO: change to variable model
modeltype = "deepfilternet2"

class ModelParams(DfParams):
    def __init__(self):
        self.__model = config("MODEL", default=modeltype, section="train")
        self.__params = getattr(import_module(self.__model), "ModelParams")()

    def __getattr__(self, attr: str):
        return getattr(self.__params, attr)


def init_model(*args, **kwargs):
    """Initialize the model specified in the config."""
    logger.info(f"Initializing model `{modeltype}`")
    model = getattr(import_module(modeltype), "init_model")(*args, **kwargs)
    model.to(memory_format=torch.channels_last)
    return model