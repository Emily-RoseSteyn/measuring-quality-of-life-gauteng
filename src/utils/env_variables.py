# env_variables.py

"""This module defines project-level env variable keys."""
import logging
import os

from dotenv import load_dotenv

from utils.enum import enum

load_dotenv()

PLANET_API_KEY = os.getenv("PLANET_API_KEY", "")
PLANET_API_URL_BASEMAPS = "https://api.planet.com/basemaps/v1/mosaics"

LOGGING_LEVEL = os.getenv("LOGGING_LEVEL", logging.INFO)

SLURM_ENABLED = os.getenv("SLURM_ENABLED", None)

MPI_TAGS = enum("READY", "DONE", "EXIT", "START")
