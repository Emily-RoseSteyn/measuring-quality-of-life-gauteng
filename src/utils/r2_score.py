import tensorflow as tf


# Custom r2 needed because tf 2.11 does not have this as a metric
# Additionally, have to use tf 2.11 because of cluster constraints
def r_squared(y_true, y_pred):
    """Custom metric function to calculate R-squared."""
    ss_res = tf.reduce_mean(tf.square(y_true - y_pred))
    ss_tot = tf.reduce_mean(tf.square(tf.math.subtract(y_true, tf.reduce_mean(y_true))))
    return 1 - ss_res / (ss_tot + tf.keras.backend.epsilon())
