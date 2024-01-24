import os

from utils.logger import get_logger


def main() -> None:
    logger = get_logger()
    logger.info("In tile images")

    # Output directory
    results_dir = "./outputs/tiles"

    if not os.path.isdir(results_dir):
        os.makedirs(results_dir)

    # Get images directory
    image_dir = "data/basemap-quads/2021"
    for image in os.listdir(image_dir):
        # check if the image ends with tiff
        if image.endswith(".tiff"):
            logger.debug(image)


if __name__ == "__main__":
    main()
