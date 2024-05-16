import json
import os
from datetime import datetime

import pandas as pd
from utils.logger import get_logger


def main() -> None:
    logger = get_logger()
    logger.info("In gcro data processing")
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
    selected_data = df_gcro[
        [
            "ward_code",
            "F1servic",
            "F2soclas",
            "F3govsat",
            "F4lifsat",
            "F5health",
            "F6safety",
            "F7partic",
            "QoLIndex_Data_Driven",
        ]
    ].rename(
        columns={
            "F1servic": "services",
            "F2soclas": "socioeconomic_status",
            "F3govsat": "government_satisfaction",
            "F4lifsat": "life_satisfaction",
            "F5health": "health",
            "F6safety": "safety",
            "F7partic": "participation",
            "QoLIndex_Data_Driven": "qol_index",
        }
    )
    ward_code_group = selected_data.groupby(["ward_code"])

    ward_code_clusters = (
        ward_code_group.size().to_frame(name="counts").join(ward_code_group.mean())
    )
    ward_code_clusters.to_csv(f"{results_dir}/gcro-clustered-data.csv")

    # TODO: Plot of distribution of interview dates


if __name__ == "__main__":
    main()
