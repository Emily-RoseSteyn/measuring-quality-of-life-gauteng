import json
from pathlib import Path

import requests
from utils.env_variables import PLANET_API_KEY, PLANET_API_URL_BASEMAPS
from utils.logger import get_logger


def initialise_session() -> requests.Session:
    # Setup the session
    session = requests.Session()

    # Authenticate
    session.auth = (PLANET_API_KEY, "")
    return session


def get_quad_url(mosaic_id: str, quad_id: str) -> str:
    return f"{PLANET_API_URL_BASEMAPS}/{mosaic_id}/quads/{quad_id}/full"


def main() -> None:
    logger = get_logger()
    logger.info("In data download")

    session = initialise_session()

    # Load metadata
    with open("data/basemap-metadata/basemap-metadata.json") as f:
        data = json.load(f)

    # 2021
    year = "2021"
    bm_2021 = data[year]
    mosaic_name = bm_2021["mosaic_name"]
    quad_ids = bm_2021["basemap_quad_ids"]

    parameters = {"name__is": mosaic_name}

    # Make get request to access mosaic from basemaps API
    res = session.get(PLANET_API_URL_BASEMAPS, params=parameters)
    mosaic = res.json()

    # Get mosaic id from response
    mosaic_id = mosaic["mosaics"][0]["id"]

    for quad_id in quad_ids:
        link = get_quad_url(mosaic_id, quad_id)
        file_path = f"data/basemap-quads/{year}/{quad_id}.tiff"

        # Check if file already exists
        file = Path(file_path)
        if file.is_file():
            continue

        r = session.get(link)
        with open(file_path, "wb") as f:
            f.write(r.content)


if __name__ == "__main__":
    main()
