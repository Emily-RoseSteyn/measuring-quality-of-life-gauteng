import tensorflow as tf
from utils.logger import get_logger


def main() -> None:
    logger = get_logger()
    logger.info("In featurization")

    # Check if running on GPU
    logger.debug("GPUs.................................. ")
    logger.debug(tf.config.list_physical_devices("GPU"))


if __name__ == "__main__":
    main()
