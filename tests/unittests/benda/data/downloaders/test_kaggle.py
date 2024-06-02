import os
import json
from pathlib import Path

import pytest

from helpers.files import TmpFile
import autoda_plugins.datasets.downloaders.kaggle as kaggle


KAGGLE_USERNAE_VAL = 'username'
KAGGLE_KEY_VAL = 'key'
FILE_PATH = Path(__file__).parent.resolve()
TMP_PATH = (FILE_PATH / '..' / '..' / '..' / '..' / 'tmp').resolve()
DATA_PATH = (FILE_PATH / '..' / '..' / '..' / '..' / 'data').resolve()


def json_dumps(content) -> str:
    return json.dumps(content, indent=4)


def create_valid_creds_file(file_name: str) -> TmpFile:
    return TmpFile(file_name, json_dumps({'username': KAGGLE_USERNAE_VAL, 'key': KAGGLE_KEY_VAL}))


def create_invalid_creds_file(file_name: str) -> TmpFile:
    return TmpFile(file_name, json_dumps({'test': 'test'}))


def test_read_kaggle_creds_invalid_file_name() -> None:
    os.environ.pop(kaggle._KAGGLE_USERNAME, None)
    os.environ.pop(kaggle._KAGGLE_KEY, None)
    assert kaggle.read_kaggle_creds(TMP_PATH / 'invalid_file_name_asdfasdf.json') is False
    assert kaggle._KAGGLE_USERNAME not in os.environ
    assert kaggle._KAGGLE_KEY not in os.environ


def test_read_kaggle_creds_invalid_file_content() -> None:
    os.environ.pop(kaggle._KAGGLE_USERNAME, None)
    os.environ.pop(kaggle._KAGGLE_KEY, None)
    file_name = TMP_PATH / 'invalid_kaggle.json'
    temp_file = create_invalid_creds_file(file_name)  # noqa: F841
    assert kaggle.read_kaggle_creds(file_name) is False
    assert kaggle._KAGGLE_USERNAME not in os.environ
    assert kaggle._KAGGLE_KEY not in os.environ


def test_read_kaggle_creds_valid_file() -> None:
    file_name = TMP_PATH / 'kaggle.json'
    temp_file = create_valid_creds_file(file_name)  # noqa: F841
    assert kaggle.read_kaggle_creds(file_name) is True
    assert os.environ[kaggle._KAGGLE_USERNAME] == KAGGLE_USERNAE_VAL
    assert os.environ[kaggle._KAGGLE_KEY] == KAGGLE_KEY_VAL


def test_read_kaggle_creds_valid_dir() -> None:
    file_name = TMP_PATH / 'kaggle.json'
    temp_file = create_valid_creds_file(file_name)  # noqa: F841
    assert kaggle.read_kaggle_creds(TMP_PATH) is True
    assert os.environ[kaggle._KAGGLE_USERNAME] == KAGGLE_USERNAE_VAL
    assert os.environ[kaggle._KAGGLE_KEY] == KAGGLE_KEY_VAL


def test_clear_kaggle_creds() -> None:
    os.environ[kaggle._KAGGLE_USERNAME] = KAGGLE_USERNAE_VAL
    os.environ[kaggle._KAGGLE_KEY] = KAGGLE_KEY_VAL
    kaggle.clear_kaggle_creds()
    assert kaggle._KAGGLE_USERNAME not in os.environ
    assert kaggle._KAGGLE_KEY not in os.environ


def test_search_for_kaggle_creds_invalid_paths() -> None:
    assert kaggle.search_for_kaggle_creds([TMP_PATH / 'path' / 'to' / 'creds']) is False


def test_search_for_kaggle_creds_valid_paths() -> None:
    file_name = TMP_PATH / 'kaggle.json'
    temp_file = create_valid_creds_file(file_name)  # noqa: F841
    assert kaggle.search_for_kaggle_creds([file_name]) is True


@pytest.fixture
def kaggle_downloader() -> kaggle.KaggleDownloader:
    return kaggle.KaggleDownloader(DATA_PATH / 'owner' / 'dataset')


@pytest.fixture
def kaggle_downloader_creds() -> kaggle.KaggleDownloader:
    return kaggle.KaggleDownloader(DATA_PATH / 'owner' / 'dataset', TMP_PATH / 'kaggle.json')


def test_init_dataset_id(kaggle_downloader) -> None:
    assert kaggle_downloader.dataset_id == DATA_PATH / 'owner' / 'dataset'


def test_init_default_creds_file(kaggle_downloader) -> None:
    assert kaggle_downloader.creds_file is None


def test_init_creds_file(kaggle_downloader_creds) -> None:
    assert kaggle_downloader_creds.creds_file == TMP_PATH / 'kaggle.json'


def test_valid_search_for_creds(kaggle_downloader) -> None:
    file_name = TMP_PATH / 'kaggle.json'
    temp_file = create_valid_creds_file(file_name)  # noqa: F841
    kaggle_downloader._default_creds_search_paths = [file_name]
    assert kaggle_downloader._search_for_creds() is True


def test_custom_path_search_for_creds(kaggle_downloader) -> None:
    file_name = TMP_PATH / 'mycreds' / 'kaggle.json'
    temp_file = create_valid_creds_file(file_name)  # noqa: F841
    kaggle_downloader._default_creds_search_paths = [TMP_PATH, TMP_PATH / 'asdf']
    kaggle_downloader.creds_file = file_name
    assert kaggle_downloader._search_for_creds() is True


def test_invalid_search_for_creds(kaggle_downloader) -> None:
    kaggle_downloader._default_creds_search_paths = [TMP_PATH / 'invalide_dir_asdf/kaggle.json']
    assert kaggle_downloader._search_for_creds() is False


def test_default_creds_search_paths(kaggle_downloader) -> None:
    assert '~/.kaggle/kaggle.json' in kaggle_downloader._default_creds_search_paths or \
        '~/.kaggle' in kaggle_downloader._default_creds_search_paths


def test_kaggle_download_invalid_creds(kaggle_downloader) -> None:
    kaggle_downloader._default_creds_search_paths = [TMP_PATH / 'invalide_dir_asdf/kaggle.json']
    assert kaggle_downloader.download() is False


def test_kaggle_download_invalid_dataset_id(kaggle_downloader) -> None:
    kaggle_downloader.dataset_id = 'invalid'
    assert kaggle_downloader.download() is False
