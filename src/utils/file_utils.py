import os
from typing import Generator, Literal

FileType = Literal["tiff", "csv"]


def dir_nested_file_list(
    directory: str, file_type: FileType
) -> Generator[str, None, None]:
    for dirpath, _, filenames in os.walk(directory):
        for f in filenames:
            if f.endswith(file_type):
                yield os.path.abspath(os.path.join(dirpath, f))
