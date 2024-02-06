from env_variables import PLANET_API_KEY


def main() -> None:
    if not PLANET_API_KEY:
        raise KeyError("Plant API Key is missing")


if __name__ == "__main__":
    main()
