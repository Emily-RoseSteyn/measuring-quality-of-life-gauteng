import logging

from utils.env_variables import LOGGING_LEVEL


def get_logger() -> logging.Logger:
    logging.basicConfig(level=LOGGING_LEVEL, format="%(asctime)s | %(levelname)s | %(message)s",
                        datefmt="%Y-%m-%d %H:%M:%S")

    return logging.getLogger("_")
