# env_variable_keys.py

"""This module defines project-level env variable keys."""
import os

PLANET_API_KEY = "PLANET_API_KEY"
LOGGING_LEVEL = "LOGGING_LEVEL"
SLURM_ENABLED = os.getenv("SLURM_ENABLED", None)
