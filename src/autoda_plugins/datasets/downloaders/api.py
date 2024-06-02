from typing import Protocol, Optional


class Downloader(Protocol):
    """
    An abstract class representing a Downloader.

    Use this class as base class if you want to implement a custom Downloader.
    """

    def download(self, target: str = '.', dataset_name: Optional[str] = None,
                 force: bool = False) -> bool:
        """
        Implement this method so that it downloads a dataset from a server of your choice.
        The dataset should be stored under `target/dataset_name`. Use `force` to check whether
        a force download should be performed.

        - `target: str = '.' The target directory for the dataset. The default value is '.'.

        - `dataset_name: str = None` A user defined name for the dataset. If it is None the
        dataset name it self should be used as `dataset_name`. The default value is None.

        - `force: bool = False` If the value is True, the download should also be executed
        if the target/dataset_name directory is not empty. The default value is False.

        The method should return True if the download was successful and False otherwise.
        """
