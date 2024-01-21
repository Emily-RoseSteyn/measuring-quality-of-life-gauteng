import logging
import os

import pandas as pd


def main() -> None:
    logging.info("In gcro data processing")
    # Load gcro data
    gcro_dta = "./data/surveys/gcro-2021.dta"
    df_gcro = pd.read_stata(gcro_dta, convert_categoricals=False)

    # TODO: Replace with step parameter
    results_dir = "./outputs/processed-gcro"

    if not os.path.isdir(results_dir):
        os.makedirs(results_dir)

    # 2021
    dtypes = df_gcro.dtypes
    dtypes.to_json(
        path_or_buf=f"{results_dir}/gcro-column-datatypes.json", default_handler=str
    )


if __name__ == "__main__":
    main()
