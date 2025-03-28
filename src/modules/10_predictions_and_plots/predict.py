import geopandas as gpd
from dvc.api import params_show
from keras.src.saving import load_model
from utils.keras_data_format import create_generator

params = params_show()


def make_predictions(df: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    label = params["train"]["label"]
    generator = create_generator(df, label)

    # Results directory where everything is
    model_file = "outputs/model/final.keras"

    # Load model
    model = load_model(model_file)

    # Make predictions
    df["actual"] = df[label]
    df["predicted"] = model.predict(generator)
    return df
