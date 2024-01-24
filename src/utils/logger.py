import logging
import os

from utils.env_variable_keys import LOGGING_LEVEL


def get_logger() -> logging.Logger:
    logging_level = os.environ.get(LOGGING_LEVEL, logging.DEBUG)
    logging.basicConfig(level=logging_level)
    return logging.getLogger("masters")
