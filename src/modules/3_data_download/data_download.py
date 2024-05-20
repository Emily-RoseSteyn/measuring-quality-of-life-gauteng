import json
import os
from pathlib import Path

import requests
from tqdm import tqdm

from utils.env_variables import PLANET_API_KEY, PLANET_API_URL_BASEMAPS
from utils.logger import get_logger

logger = get_logger()


def initialise_session() -> requests.Session:
    # Setup the session
    session = requests.Session()

    # Authenticate
    session.auth = (PLANET_API_KEY, "")
    return session


def get_quad_url(mosaic_id: str, quad_id: str) -> str:
    return f"{PLANET_API_URL_BASEMAPS}/{mosaic_id}/quads/{quad_id}/full"


def download_satellite_basemaps(key: str, mosaic_name: str, quad_ids):
    logger.info(f"Downloading data for {key}")
    session = initialise_session()
    parameters = {"name__is": mosaic_name}

    # Make get request to access mosaic from basemaps API
    res = session.get(PLANET_API_URL_BASEMAPS, params=parameters)
    mosaic = res.json()

    # Get mosaic id from response
    mosaic_id = mosaic["mosaics"][0]["id"]

    # Create results dir
    results_dir = f"data/basemap-quads/{key}"
    if not os.path.isdir(results_dir):
        os.makedirs(results_dir)

    # For each quad id, download data if it doesn't already exist
    for quad_id in tqdm(quad_ids):
        link = get_quad_url(mosaic_id, quad_id)
        file_path = f"{results_dir}/{quad_id}.tiff"

        # Check if file already exists
        file = Path(file_path)
        if file.is_file():
            continue

        r = session.get(link)
        with open(file_path, "wb") as f:
            f.write(r.content)


def main() -> None:
    logger.info("In data download")

    # Load metadata
    with open("data/basemap-metadata/basemap-metadata.json") as f:
        basemap_metadata = json.load(f)

    # Get keys in basemaps metadata
    keys = basemap_metadata.keys()

    for key in keys:
        bm = basemap_metadata[key]
        mosaic_name = bm["mosaic_name"]
        quad_ids = bm["basemap_quad_ids"]
        date = bm["date"]
        download_satellite_basemaps(date, mosaic_name, quad_ids)
        # TODO: Need some kind of output here


if __name__ == "__main__":
    main()
