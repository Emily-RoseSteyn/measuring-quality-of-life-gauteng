import json
import logging
import os
from datetime import datetime

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

    # Write column headings and data types to json
    dtypes = df_gcro.dtypes
    dtypes.to_json(
        path_or_buf=f"{results_dir}/gcro-column-datatypes.json", default_handler=str
    )

    # Write date information to json
    date_format = "%Y-%m-%d"
    total_interviews = len(df_gcro)
    min_date = datetime.strftime(min(df_gcro["interview_date"]), date_format)
    max_date = datetime.strftime(max(df_gcro["interview_date"]), date_format)
    mean_date = datetime.strftime(df_gcro["interview_date"].mean(), date_format)

    date_info = {
        "total_interviews": total_interviews,
        "min_date": min_date,
        "max_date": max_date,
        "mean_date": mean_date,
    }

    with open(f"{results_dir}/gcro-date-info.json", "w") as f:
        json.dump(date_info, f, default=str)

    # Ward Code Clusters
    # TODO: Plot of distribution of interview dates


if __name__ == "__main__":
    main()
