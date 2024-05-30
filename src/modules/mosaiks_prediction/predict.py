import os
from typing import Any

import pandas as pd
import tensorflow as tf
from pandas import merge, read_csv
from plotnine import (
    aes,
    geom_abline,
    geom_point,
    ggplot,
    ggtitle,
    labs,
    scale_color_cmap,
)
from sklearn import metrics
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.linear_model._base import LinearModel
from sklearn.model_selection import train_test_split
from utils.logger import get_logger

ModelResult = tuple[str, Any, Any, float | Any]

logger = get_logger()


def main() -> None:
    # TODO: SPLIT THIS UP
    logger.info("In prediction")

    # Tensorflow info logs
    logger.info("TensorFlow version: %s", tf.__version__)
    device_name = tf.test.gpu_device_name()
    if device_name != "/device:GPU:0":
        logger.info("GPU device not found - On for CPU time!")
    else:
        logger.info("Found GPU at %s", device_name)

    # Load mosaik features (this may take a few mins)
    mosaik_features = read_csv("data/mosaiks-features/mosaiks_features.csv")
    mosaik_features = mosaik_features.sort_values(by=["Lat", "Lon"])
    logger.debug(mosaik_features.head())

    # Load label data
    label_data = read_csv("outputs/grid/qol-labelled-grid.csv")
    label_data = label_data.sort_values(by=["latitude", "longitude"])
    logger.debug(label_data.head())

    # Merge features with label
    merged_labelled_features = merge(
        label_data,
        mosaik_features,
        how="left",
        left_on=["latitude", "longitude"],
        right_on=["Lat", "Lon"],
    )

    logger.info("%s observations in merged dataframe", len(merged_labelled_features))

    # Remove invalid rows
    cleaned_labelled_features = merged_labelled_features.dropna()
    only_na = merged_labelled_features[
        ~merged_labelled_features.index.isin(cleaned_labelled_features.index)
    ]

    logger.info("%s rows with NAs dropped.", len(only_na))
    logger.info("%s observations in final dataframe.", len(cleaned_labelled_features))

    # Ridge Regression
    # TODO: fixed effects

    # Set label attribute name in the dataframe
    # TODO: Later, modify according to experiment
    label_value = "qol_index"

    # Split data into training and test sets
    # TODO: Parameterise constants
    random_state_seed = 42
    test_size = 0.2
    features = cleaned_labelled_features.drop(
        [
            label_value,
            "index_right",
            "ward_code",
            "counts",
            "services",
            "socioeconomic_status",
            "government_satisfaction",
            "life_satisfaction",
            "health",
            "safety",
            "participation",
        ],
        axis=1,
    )
    labels = cleaned_labelled_features[label_value]
    x_train, x_test, y_train, y_test = train_test_split(
        features, labels, test_size=test_size, random_state=random_state_seed
    )

    # Saving geometry
    plotting_coords = x_test[["longitude", "latitude"]]
    plotting_coords_train = x_train[["longitude", "latitude"]]

    # Dropping geometry
    x_train = x_train.drop(
        [
            "latitude",
            "longitude",
            "geometry",
            "Lat",
            "Lon",
        ],
        axis=1,
    )
    x_test = x_test.drop(
        [
            "latitude",
            "longitude",
            "geometry",
            "Lat",
            "Lon",
        ],
        axis=1,
    )

    # OLS
    ols_result = ols_regression(x_test, x_train, y_test, y_train)
    plot_results(plotting_coords, plotting_coords_train, y_test, y_train, ols_result)

    # RIDGE
    ridge_result = ridge_regression(x_test, x_train, y_test, y_train)
    plot_results(plotting_coords, plotting_coords_train, y_test, y_train, ridge_result)


def regress(reg: LinearModel, x_test: dict, x_train: dict, y_train: dict) -> tuple:
    # Train the model
    reg.fit(x_train, y_train)
    # Make predictions
    y_pred = reg.predict(x_test)
    # Make predictions on training set for plotting maps
    y_pred_train = reg.predict(x_train)
    y_pred_train[y_pred_train < 0] = 0
    return y_pred, y_pred_train


def calc_r2(model_type: str, y_test: dict, y_pred: dict) -> Any:
    # Compute R^2 from true and predicted values
    r2 = metrics.r2_score(y_test, y_pred)
    logger.info("%s r2: %s", model_type, r2)
    return r2


def ols_regression(
    x_test: dict, x_train: dict, y_test: dict, y_train: dict
) -> ModelResult:
    model_type = "ols"
    reg = LinearRegression()
    y_pred, y_pred_train = regress(reg, x_test, x_train, y_train)
    r2 = calc_r2(model_type, y_test, y_pred)
    return model_type, y_pred, y_pred_train, r2


def ridge_regression(
    x_test: dict, x_train: dict, y_test: dict, y_train: dict
) -> ModelResult:
    model_type = "ridge"
    reg = Ridge(alpha=1)  # alpha is the hyperparameter equivalent to lambda
    y_pred, y_pred_train = regress(reg, x_test, x_train, y_train)
    r2 = calc_r2(model_type, y_test, y_pred)
    return model_type, y_pred, y_pred_train, r2


def plot_results(
    plotting_coords: dict,
    plotting_coords_train: dict,
    y_test: dict,
    y_train: dict,
    result: ModelResult,
) -> None:
    model_type, y_pred, y_pred_train, r2 = result

    # Plots
    results_dir = "./outputs/mosaiks-prediction"

    if not os.path.isdir(results_dir):
        os.makedirs(results_dir)
    map_plot = pd.DataFrame(
        {
            "longitude": plotting_coords["longitude"],
            "latitude": plotting_coords["latitude"],
            "predicted": y_pred,
            "observed": y_test,
        }
    )
    # Scatterplot observed vs predicted
    fig = (
        ggplot(map_plot, aes(x=y_pred, y=y_test))
        + geom_point(alpha=0.5)
        + geom_abline(intercept=0, slope=1, size=0.8, alpha=0.3)
        + labs(x="Predicted", y="Observed")
        + ggtitle(f"Predicted vs Observed: r2 = {round(r2, 4)}")
    )
    fig.save(f"{results_dir}/{model_type}-observed-vs-predicted.png", dpi=300)
    map_plot_train = pd.DataFrame(
        {
            "longitude": plotting_coords_train["longitude"],
            "latitude": plotting_coords_train["latitude"],
            "predicted": y_pred_train,
            "observed": y_train,
        }
    )
    # Plot observed
    # noinspection PyTypeChecker
    #   - Warning is incorrect
    observed = (
        ggplot()
        + geom_point(
            data=map_plot_train,
            mapping=aes(x="longitude", y="latitude", color="observed"),
            size=0.3,
            alpha=0.8,
        )
        + scale_color_cmap(cmap_name="Spectral")
        + labs(x="Longitude", y="Latitude", title="Observed", color="QoL")
    )
    observed.save(f"{results_dir}/{model_type}-observed.png", dpi=300)
    # Plot predicted
    # noinspection PyTypeChecker
    #   - Warning is incorrect
    predicted = (
        ggplot()
        + geom_point(
            data=map_plot_train,
            mapping=aes(x="longitude", y="latitude", color="predicted"),
            size=0.3,
            alpha=0.8,
        )
        + scale_color_cmap(cmap_name="Spectral")
        + labs(x="Longitude", y="Latitude", title="Predicted", color="QoL")
    )
    predicted.save(f"{results_dir}/{model_type}-predicted.png", dpi=300)


if __name__ == "__main__":
    main()
