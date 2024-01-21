import logging

import pandas as pd


def main() -> None:
    logging.info("In gcro data processing")
    # Load gcro data
    gcro_dta = "./data/surveys/gcro-2021.dta"
    df_gcro = pd.read_stata(gcro_dta, convert_categoricals=False)
    logging.info(df_gcro.head())


if __name__ == "__main__":
    main()
