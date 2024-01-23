# TAKEN FROM https://github.com/microsoft/PlanetaryComputerExamples/blob/main/tutorials/mosaiks.ipynb
# type: ignore  # noqa: PGH003
# TODO: Fix the typings in this file
import tensorflow as tf
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


class RCF(tf.keras.layers.Layer):
    """A model for extracting Random Convolution Features (RCF) from input imagery."""

    def __init__(self, num_features=16, kernel_size=3, num_input_channels=3):
        super().__init__()

        # We create `num_features / 2` filters so require `num_features` to be divisible by 2
        assert num_features % 2 == 0

        self.conv1 = Conv2D(
            num_input_channels,
            num_features // 2,
            kernel_size=kernel_size,
            stride=1,
            padding=0,
            dilation=1,
            bias=True,
        )

        tf.keras.init.normal_(self.conv1.weight, mean=0.0, std=1.0)
        tf.keras.init.constant_(self.conv1.bias, -1.0)

    def forward(self, x):
        x1a = ReLU(self.conv1(x), inplace=True)
        x1b = ReLU(-self.conv1(x), inplace=True)

        x1a = AvgPool2D(x1a, (1, 1)).squeeze()
        x1b = AvgPool2D(x1b, (1, 1)).squeeze()

        if len(x1a.shape) == 1:  # case where we passed a single input
            return tf.cat((x1a, x1b), dim=0)
        return tf.cat((x1a, x1b), dim=1)
