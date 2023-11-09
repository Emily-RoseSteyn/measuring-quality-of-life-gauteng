import os

from dotenv import load_dotenv
from env_variable_keys import PLANET_API_KEY


def main() -> None:
    load_dotenv()  # take environment variables from .env.
    if PLANET_API_KEY in os.environ:
        pass


if __name__ == "__main__":
    main()
