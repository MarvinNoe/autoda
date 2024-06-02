from pathlib import Path


class TmpFile():
    """
    Helper class for creating a temporary file.
    It uses the RAII pattern. This means that the file is created when an object
    is initialized and deleted when the object is deleted.
    """

    def __init__(self, file_path: str, content: str) -> None:
        path = Path(file_path)

        if not path.exists():
            path.parent.mkdir(parents=True, exist_ok=True)

        self._path = path

        with open(path, 'w') as f:
            f.write(content)

    def __del__(self) -> None:
        if self._path.exists():
            self._path.unlink()

    @property
    def file_path(self) -> str:
        return str(self._path)
