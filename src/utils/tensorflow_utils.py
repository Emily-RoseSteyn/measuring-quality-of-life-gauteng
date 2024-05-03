import tensorflow as tf

from utils.logger import get_logger


def log_tf_gpu():
    logger = get_logger()
    logger.info(f"TensorFlow version: {tf.__version__}")
    device_name = tf.test.gpu_device_name()
    if device_name != "/device:GPU:0":
        logger.info("GPU device not found - On for CPU time!")
    else:
        logger.info(f"Found GPU at {device_name}")
