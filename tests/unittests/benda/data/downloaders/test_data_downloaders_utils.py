import typing
import typing_extensions

from pathlib import Path

from helpers.files import TmpFile

import autoda_plugins.datasets.downloaders.utils as utils
import autoda_plugins.datasets.downloaders.api as api

FILE_PATH = Path(__file__).parent.resolve()
TMP_PATH = (FILE_PATH / '..' / '..' / '..' / '..' / 'tmp').resolve()


def test_data_exists() -> None:
    path = TMP_PATH / 'my' / 'data' / 'file.txt'
    txt_file = TmpFile(path, 'data file')  # noqa: F841
    assert utils.data_exists(path.parent)


def test_no_data_exists() -> None:
    path = TMP_PATH / 'my' / 'utils' / 'test'
    assert utils.data_exists(path) is False


def test_downloader_is_protocol() -> None:
    assert isinstance(api.Downloader, type)
    assert getattr(api.Downloader, '_is_protocol', False)
    assert api.Downloader is not typing.Protocol
    assert api.Downloader is not typing_extensions.Protocol


def test_downloader_download() -> None:
    assert hasattr(api.Downloader, 'download')

    class Dummy(api.Downloader):
        pass

    d = Dummy()

    assert d.download() is None
