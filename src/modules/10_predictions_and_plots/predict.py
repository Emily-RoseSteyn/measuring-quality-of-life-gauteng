import geopandas as gpd
from dvc.api import params_show
from tensorflow.keras.models import load_model

from utils import custom_r_squared
from utils.keras_data_format import create_generator

params = params_show()


def make_predictions(df: gpd.GeoDataFrame):
    label = params["train"]["label"]
    generator = create_generator(df, label)

    # Results directory where everything is
    model_file = "outputs/model/final.h5"

    # Load model
    model = load_model(model_file, custom_objects={"custom_r_squared": custom_r_squared})

    # Make predictions
    df["actual"] = df[label]
    df["predicted"] = model.predict(generator)
    return df
