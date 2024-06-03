import os
from pathlib import Path

import pandas as pd
from keras_preprocessing.image import ImageDataGenerator
from matplotlib import pyplot as plt


def create_generator(
        df: pd.DataFrame,
        label: str,
        apply_augmentation_flag: int = 0,
        visualize_augmentations_flag: int = 0
):
    """
    Adapted from https://rosenfelder.ai/keras-regression-efficient-net/
    Accepts a Pandas DataFrames. Creates and returns a keras ImageDataGenerator.
    The augmentations of the ImageDataGenerators can also be visualised in this function.

    Parameters
    ----------
    df : pd.DataFrame
        Pandas DataFrame containing all data.
    label : str
        The label being used as the target feature
    apply_augmentation_flag: int
        Flag to apply augmentations.
    visualize_augmentations_flag: int
        Flag to visualise augmentations.

    Returns
    -------
    ImageDataGenerator
        keras ImageDataGenerator.
    """
    # Create training ImageDataGenerator with image augmentations

    generator = ImageDataGenerator(
        rescale=1.0 / 255
    )

    if apply_augmentation_flag:
        generator.horizontal_flip = True
        generator.vertical_flip = True

        # TODO: Augmentations
        #  Check if following necessary or any others needed?
        # generator.width_shift_range = 0.1
        # generator.height_shift_range = 0.1
        # generator.brightness_range = (0.75, 1)
        # generator.shear_range = 0.1
        # generator.zoom_range = [0.75, 1]

    # Visualize image augmentations if flag set before actually getting dataframe
    if visualize_augmentations_flag == 1:
        visualize_augmentations(generator, df)

    # Create dataframe iterator
    generator.flow_from_dataframe(
        directory="outputs/tiles",
        dataframe=df,
        x_col="tile",  # Image location
        y_col=label,  # Target feature
        class_mode="raw",  # Use "raw" for regressions TODO: Understand why?
        target_size=(256, 256),
        # TODO: increase or decrease to fit GPU
        batch_size=32,
    )
    return generator


# TODO: Check if this is right place
def visualize_augmentations(data_generator: ImageDataGenerator, df: pd.DataFrame):
    """
    Adapted from https://rosenfelder.ai/keras-regression-efficient-net/
    Visualizes the keras augmentations with matplotlib in 3x3 grid. This function is part of create_generators() and
    can be accessed from there.

    Parameters
    ----------
    data_generator : Iterator
        The keras data generator of your training data.
    df : pd.DataFrame
        The Pandas DataFrame containing your training data.
    """
    image_dir = os.path.abspath(Path("./outputs/misc"))

    if not os.path.isdir(image_dir):
        os.makedirs(image_dir)

    # super hacky way of creating a small dataframe with one image
    series = df.iloc[2]

    df_augmentation_visualization = pd.concat([series, series], axis=1).transpose()

    iterator_visualizations = data_generator.flow_from_dataframe(
        directory="outputs/tiles",
        dataframe=df_augmentation_visualization,
        x_col="tile",
        y_col="qol_index",
        class_mode="raw",
        target_size=(256, 256),
        batch_size=1,  # use only one image for visualization
    )

    for i in range(9):
        plt.subplot(3, 3, i + 1)  # create a 3x3 grid
        batch = next(iterator_visualizations)  # get the next image of the generator (always the same image)
        img = batch[0]
        img = img[0, :, :, :]  # remove one dimension for plotting without issues
        plt.imshow(img)
    plt.savefig("outputs/misc/augmentations.png")
