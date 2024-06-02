from pathlib import Path


def data_exists(dir_path: Path) -> bool:
    """
    Checks whether the specified directory is empty.

    - `dir_path: Path` Path to the directory.

    Returns True if the directory in not empty, False otherwise.
    """
    if dir_path.is_dir():
        if any(dir_path.iterdir()):
            return True
    return False
