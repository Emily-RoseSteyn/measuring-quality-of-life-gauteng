import tensorflow as tf
from utils.logger import get_logger


def main() -> None:
    logger = get_logger()
    logger.info("In featurization")

    # Tensorflow info logs
    logger.debug("TensorFlow version: %s", tf.__version__)
    device_name = tf.test.gpu_device_name()
    if device_name != "/device:GPU:0":
        logger.debug("GPU device not found - On for CPU time!")
    else:
        logger.debug("Found GPU at %s", device_name)


if __name__ == "__main__":
    main()
