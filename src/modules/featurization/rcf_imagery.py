# ADAPTED FROM https://github.com/microsoft/PlanetaryComputerExamples/blob/main/tutorials/mosaiks.ipynb
# type: ignore  # noqa: PGH003
# TODO: Fix the typings in this file
import tensorflow as tf
from keras import Model
from tensorflow.keras.layers import AvgPool2D, Conv2D, ReLU


def featurize(input_img, model, device):
    """Helper method for running an image patch through the model.

    Args:
        :param input_img: (np.ndarray): Image in (C x H x W) format with a dtype of uint8.
        :param model: (torch.nn.Module): Feature extractor network
        :param device:
    """
    expected_shape = 3
    assert len(input_img.shape) == expected_shape
    input_img = tf.from_numpy(input_img / 255.0).float()
    input_img = input_img.to(device)
    with tf.no_grad():
        feats = model(input_img.unsqueeze(0)).cpu().numpy()
    return feats


class RCF(Model):
    """A model for extracting Random Convolution Features (RCF) from input imagery."""

    def __init__(self, num_features=16, kernel_size=3, num_input_channels=3):
        super().__init__()

        # We create `num_features / 2` filters so require `num_features` to be divisible by 2
        assert num_features % 2 == 0

        self.conv1 = Conv2D(
            filters=num_features // 2,  # Number of output filters
            kernel_size=kernel_size,  # Kernel size
            strides=(1, 1),  # Strides
            padding="valid",  # Padding, equivalent to padding=0 in PyTorch
            dilation_rate=(1, 1),  # Dilation rate
            use_bias=True,  # Whether to use a bias term
            input_shape=(None, None, num_input_channels),  # Input shape (optional)
        )

        # Create a normal initializer with mean 0 and standard deviation 1:
        initializer = tf.random_normal_initializer(mean=0.0, stddev=1.0)

        # Apply the initializer to the weights of the 'conv1' layer:
        self.conv1.build(
            input_shape=(None, None, num_input_channels)
        )  # Ensure weights are created
        self.conv1.weight = initializer

        # Create a constant initializer with value -1.0:
        initializer = tf.constant_initializer(-1.0)

        # Apply the initializer to the bias of the 'conv1' layer:
        self.conv1.build(
            input_shape=(None, None, num_input_channels)
        )  # Ensure biases are created
        self.conv1.bias = initializer

    def forward(self, x):
        x1a = ReLU(self.conv1(x), inplace=True)
        x1b = ReLU(-self.conv1(x), inplace=True)

        x1a = AvgPool2D(x1a, (1, 1)).squeeze()
        x1b = AvgPool2D(x1b, (1, 1)).squeeze()

        if len(x1a.shape) == 1:  # case where we passed a single input
            return tf.cat((x1a, x1b), dim=0)
        return tf.cat((x1a, x1b), dim=1)
