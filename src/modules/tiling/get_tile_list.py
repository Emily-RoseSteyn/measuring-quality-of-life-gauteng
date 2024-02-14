from utils.file_utils import dir_nested_file_list


def get_tile_list() -> list:
    # Get images directory
    image_dir = "data/basemap-quads"
    return list(dir_nested_file_list(image_dir, "tiff"))
