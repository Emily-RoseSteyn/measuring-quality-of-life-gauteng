from utils.logger import get_logger


def tile_image(file_path: str, output_dir: str) -> None:
    logger = get_logger()
    logger.info("Tiling %s", file_path)
    logger.info("Output dir %s", output_dir)
