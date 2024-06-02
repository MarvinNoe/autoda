"""
Kaggle downloader module.

This module provides the basic functionality to authenticate and download datasets from Kaggle.
It uses the Kaggle API to interact with Kaggle.

"""
import os
import json
import logging
import urllib3
from pathlib import Path
from typing import Optional

from .api import Downloader
from .utils import data_exists


_LOG = logging.getLogger(__name__)
"""Module-level logger."""

_KAGGLE_USERNAME = 'KAGGLE_USERNAME'
"""Name of the environment variable that stores the Kaggle user name."""

_KAGGLE_KEY = 'KAGGLE_KEY'
"""Name of the environment variable that stores the Kaggle key."""


def read_kaggle_creds(file_path: str) -> bool:
    """
    Reads the Kaggle credentials (username, key) from the specified json file and
    registers them as environment variables.

    - `file_path: str` Path to the JSON file containing the Kaggle credentials.

    Returns True if the JSON file can be opened and the Kaggle credentials can
    be registered as environment variables. Otherwise False.
    """
    try:
        path = Path(file_path).expanduser()

        if path.is_dir():
            path = path / 'kaggle.json'

        with open(path, 'r') as f:
            data = json.load(f)
            if 'username' in data and 'key' in data:
                os.environ[_KAGGLE_USERNAME] = data['username']
                os.environ[_KAGGLE_KEY] = data['key']
                return True
    except Exception:
        return False
    return False


def clear_kaggle_creds() -> None:
    """
    Removes the Kaggle credentials (username, key) from the environment variables.
    """
    os.environ.pop(_KAGGLE_USERNAME, None)
    os.environ.pop(_KAGGLE_KEY, None)


def search_for_kaggle_creds(creds_file_paths: list[str]) -> bool:
    """
    Searches for the file with the Kaggle credentials in the specified paths.

    - `creds_file_paths: list[str]` possible paths to the JSON file containing the
    Kaggle credentials.
    """
    for creds_file in creds_file_paths:
        if read_kaggle_creds(creds_file):
            return True
    return False


class KaggleDownloader(Downloader):
    """
    Instances of the KaggleDownloader can be used to download datasets from Kaggle.com.

    Implements the .utils.Downloader abstract base class.
    """

    _default_creds_search_paths = [
        '~/.kaggle/kaggle.json',
        './kaggle.json'
    ]
    """Possible file locations for searching for Kaggle credentials (kaglle.json)."""

    def __init__(self, dataset_id: str, creds_file: Optional[str] = None) -> None:
        """
        Initializes a KaggleDownloader object.

        - `dataset_id: str` Id of the dataset to be downloaded (owner/dataset[/version]).

        - `creds_file: str = None` Optional argument to specify the path to the Kaggle
            credentials file. By default, the KaggleDownloader assumes that the credentials file
            is located under ~/.kaggle/kaggle.json or ./kaggle.json. If a creds_file is
            specified, the credentials file is also searched for in this location.
            The default value is None.
        """
        super(KaggleDownloader, self).__init__()
        self.dataset_id = dataset_id
        self.creds_file = creds_file

    @property
    def dataset_id(self) -> str:
        """
        Id of the dataset to be downloaded (owner/dataset_name[/version]).
        """
        return self._dataset_id

    @dataset_id.setter
    def dataset_id(self, value: str) -> None:
        """
        Sets the id of the dataset to be downloaded to the specified value.
        """
        self._dataset_id = value

    @property
    def creds_file(self) -> Optional[str]:
        """
        Path to the Kaggle credentials file.
        """
        return self._creds_file

    @creds_file.setter
    def creds_file(self, value: Optional[str]) -> None:
        """
        Sets the path to the Kaggle credentials file to the specified value.
        """
        self._creds_file = value

    def _search_for_creds(self) -> bool:
        creds_file_list = self._default_creds_search_paths

        if self.creds_file:
            creds_file_list.insert(0, self.creds_file)

        return search_for_kaggle_creds(creds_file_list)

    def download(self, target: str = '.', dataset_name: Optional[str] = None,
                 force: bool = False) -> bool:
        """
        Checks whether the Kaggle credentials can be found and whether they are valid.
        It then validates the specified `dataset_id` and checks whether the directory
        target/dataset_name is empty. If these conditions are met, it attempts to
        download the dataset from Kaggle to target/dataset_name.
        If the download is successful, True is returned. Otherwise False.

        - `target: str = '.'` Path to the directory in which the dataset directory should be
        created. By default the dataset directroy is created in the current working directory.

        - `dataset_name: str = None` Name of the dataset directory. If it is None,
        the directory name corresponds to the name of the dataset in Kaggle.

        - `force: bool = False` Specifies whether the dataset should also be downloded if the
        destination directory is not empty. If True, the data in the destination directory are
        deleted before the download.

        Returns True if the download was successful.
        """
        # Register Kaggle credentails as environment variables
        if not self._search_for_creds():
            _LOG.error(' The download of \'%s\' has failed. '
                       'Could not find kaggle.json!\n'
                       'You can set the path to your credentials file via the '
                       'creds_file variable of %s or '
                       'use the default location for the kaggle.json file.\n'
                       'To learn more about the Kaggle credentials: '
                       'https://www.kaggle.com/docs/api#authentication.',
                       self.dataset_id, self.__class__.__name__)
            return False

        # Kaggle authentication
        from kaggle import api as kaggle_api
        from kaggle import rest as kaggle_rest
        kaggle_api.authenticate()
        clear_kaggle_creds()

        # Validate dataset id
        try:
            owner, dataset, version_number = kaggle_api.split_dataset_string(self.dataset_id)
        except ValueError:
            _LOG.error('Download of \'%s\' skiped. '
                       'Invalid dataset id. '
                       'The id should be in format owner/dataset-name[/version]',
                       self.dataset_id)
            return False

        # Check whether the destination is empty
        target_dir_path = Path(target)

        if dataset_name is None:
            target_dir_path = target_dir_path / dataset
        else:
            target_dir_path = target_dir_path / dataset_name

        if not force and data_exists(target_dir_path):
            _LOG.info('Download of \'%s\' skiped. '
                      '%s ist not empty. '
                      'Use force=True to force download.',
                      self.dataset_id, target_dir_path)
            return False

        try:
            kaggle_api.dataset_download_files(
                dataset=self.dataset_id,
                path=target_dir_path,
                force=force,
                quiet=False,
                unzip=True
            )
        except (urllib3.exceptions.MaxRetryError, kaggle_rest.ApiException, KeyError):
            _LOG.error(' The download of \'%s\' has failed. '
                       'Check your internet connection, the dataset id and the Kaggle '
                       'credentials setup! Learn more: '
                       'https://www.kaggle.com/docs/api#authentication',
                       self.dataset_id)
            return False

        return True
