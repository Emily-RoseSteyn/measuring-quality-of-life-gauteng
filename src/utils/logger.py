import logging
import os

from dotenv import load_dotenv

from utils.env_variable_keys import LOGGING_LEVEL

load_dotenv()


def get_logger() -> logging.Logger:
    logging_level = os.environ.get(LOGGING_LEVEL, logging.INFO)
    logging.basicConfig(level=logging_level)
    return logging.getLogger("masters")
