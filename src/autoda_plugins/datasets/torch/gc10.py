
import defusedxml.ElementTree as ElementTree
import logging


from pathlib import Path
from PIL import Image
from typing import List, Optional, Callable, Any, Union, Dict

import torch

from .vision_dataset import AutoDaVisionDataset, FilePathsType
from ..downloaders.api import Downloader
from ..downloaders.kaggle import KaggleDownloader


_LOG = logging.getLogger(__name__)
"""Module-level logger."""

GC10DET_LABEL_MAP = {
    '__background__': '__background__',
    '1_chongkong': 'punching_hole',
    '2_hanfeng': 'welding_line',
    '3_yueyawan': 'crescent_gap',
    '4_shuiban': 'water_spot',
    '5_youban': 'oil_spot',
    '6_siban': 'silk_spot',
    '7_yiwu': 'inclusion',
    '8_yahen': 'rolled_pit',
    '9_zhehen': 'crease',
    '10_yaozhed': 'waist_folding',
}
"""Mapping of the Chinese labels to the English labels of the GC10-DET (v1) dataset."""

GC10DET_CLASSES = list(GC10DET_LABEL_MAP.values())


def _gc10det_v1_load_paths(dataset_path: str) -> FilePathsType:
    """
    Searches for all *.jpg files under the specified dataset_path. Then it searches for the
    corresponding target file and saves both paths relative to the dataset_path in separate lists.
    It ensures that the target at index x corresponds to the image at index x.

    This method assumes that the dataset is structured according to the GC10-DET (v1) Kaggle
    dataset provided by the user zhangyunsheng.

    https://www.kaggle.com/datasets/zhangyunsheng/defects-class-and-location

    ├── gc10det
    │   ├── images
    │   │   ├── images
    │   │   │   ├── crease
    │   │   │   │   ├── *.jpg
    │   │   │   ├── crescent_gap
    │   │   │   │   ├── *.jpg
    │   │   │   ├── ...
    │   ├── label
    │   │   ├── label
    │   │   │   ├── *.xml

    - `dataset_path: str` Path of the GC10-DET dataset.

    Returns the lists of GC10-DET (v1) image and target paths.
    """
    img_paths = []
    target_paths = []
    path = Path(dataset_path)

    for img_file in path.rglob('*.jpg'):
        target_file = path / 'label' / 'label' / f'{img_file.stem}.xml'

        if target_file.exists():
            img_paths.append(str(img_file.relative_to(path)))
            target_paths.append(str(target_file.relative_to(path)))

    return img_paths, target_paths


def _gc10det_v1_load_targets(target_path: str, classes: List[str]) -> Dict[str, Any]:
    """
    Loads the targets of the GC10-DET (v1) Kaggle dataset provided by the user zhangyunsheng.
    The target_path must be a path to an xml file with the following structure:

    <annotation>
        ...
        <object>
            ...
            <name>3_yueyawan</name>
            <bndbox>
                <xmin>62</xmin>
                <ymin>198</ymin>
                <xmax>558</xmax>
                <ymax>1000</ymax>
            </bndbox>
        </object>
        ...
    </annotation>

    where the name of the object is the label of the bndbox and the bndbox contains
    the coordinates of the bounding box. There may be several object elements.

    - `target_path: str` Path to the xml file containing the target information.

    - `classes: List[str]` List of possible labels.

    Returns a dict where *boxes* specifies a list of all bounding box
    coordinates (xmin, ymin, xmax, ymax) and *labels* specifies a list of the corresponding labels.
    Both lists are converted to troch tensors.
    """
    boxes = []
    labels = []

    tree = ElementTree.parse(target_path)
    root = tree.getroot()

    def get_coordinate(parent: Any, tag: str) -> Union[int, None]:
        if (child := parent.find(tag)) is not None and child.text is not None:
            return int(child.text)
        return None

    for member in root.findall('object'):

        if (name := member.find('name')) is not None and \
                (bndbox := member.find('bndbox')) is not None:

            # left corner x-coordinates
            xmin = get_coordinate(bndbox, 'xmin')
            # left corner y-coordinates
            ymin = get_coordinate(bndbox, 'ymin')
            # right corner x-coordinates
            xmax = get_coordinate(bndbox, 'xmax')
            # right corner y-coordinates.
            ymax = get_coordinate(bndbox, 'ymax')

            box = [xmin, ymin, xmax, ymax]

            # handling typos in labeling
            if name.text == '10_yaozhe':
                name.text = '10_yaozhed'

            if all(item is not None for item in box) and name.text in classes:
                boxes.append(box)
                labels.append(classes.index(name.text))
            else:
                _LOG.warning(
                    'Label or bndbox coordinates are invalid in XML in file: %s', target_path)
        else:
            _LOG.warning(
                'Failed to find "name" or "bndbox" element in XML in file: %s', target_path)

    return {
        'boxes':  torch.as_tensor(boxes, dtype=torch.float32),
        'labels': torch.as_tensor(labels, dtype=torch.int64)
    }


class GC10DET(AutoDaVisionDataset):
    """
    GC10-DET dataset.

    It will be downloaded from Kaggle.com
    (https://www.kaggle.com/datasets/zhangyunsheng/defects-class-and-location).

    Note that if the version of the dataset or the download source changes, the structure
    of the dataset files may also change. In this case, the private class variables of GC10DET
    may need to be adjusted. They contain the functions that depend on the structure of the
    dataset files.
    """

    _dataset_dir = 'gc10det'
    """Name of the directory in which the GC10-DET dataset is saved."""

    _downloader: Downloader = KaggleDownloader('zhangyunsheng/defects-class-and-location/1')
    """Helper that downloads the GC10-DET dataset."""

    _file_paths_reader: Callable[..., FilePathsType] = _gc10det_v1_load_paths
    """Function that reads the GC10-DET image and target paths."""

    _target_loader: Callable[..., Dict[str, Any]] = _gc10det_v1_load_targets
    """Function that loads the GC10-DET targets."""

    _class_map = GC10DET_LABEL_MAP
    """Mapping of the Chinese classes to the English classes."""

    def __init__(
            self,
            root: str,
            transforms: Optional[Callable[..., Any]] = None,
            transform: Optional[Callable[..., Any]] = None,
            target_transform: Optional[Callable[..., Any]] = None,
            use_transforms_v2: bool = True,
            download: bool = False
    ) -> None:
        """
        Initializes the GC10DET object.

        - `root: str` Root directory of dataset where gc10det exists or will be saved to
            if download is set to True.

        - `transforms: Callable = None` A function/transforms that takes in an image and a target
            and returns the transformed versions of both.

        - `transform: Callable = None` A function/transform that takes in a PIL image
            and returns a transformed version.

        - `target_transform: Callable = None` A function/transform that takes in the target and
            transforms it.

        - `use_transforms_v2: bool = True` If true, the image and the target are converted so that
            Torchvisions transforms v2 can be applied to them.

        - `convert_to_transforms_v2: bool = True` If true, the target is converted to be compatible
            with Torchvision Transforms v2. To find out more about this, see:
            https://pytorch.org/vision/stable/auto_examples/transforms/plot_transforms_getting_started.html#getting-started-with-transforms-v2

        - `download: bool = False` If true, downloads the dataset from the internet and puts it in
            root directory. If dataset is already downloaded, it is not downloaded again.
        """
        super().__init__(
            root,
            GC10DET._dataset_dir,
            transforms,
            transform,
            target_transform,
            use_transforms_v2
        )

        if download:
            self.download()

    def load_image(self, index: int) -> Image.Image:
        return Image.open(self.image_path(index))

    def load_target(self, index: int) -> Dict[str, Any]:
        target = GC10DET._target_loader(self.target_path(index), list(GC10DET._class_map.keys()))
        target['image_id'] = torch.tensor([index])
        return target

    def file_paths(self, dataset_path: str) -> FilePathsType:
        return GC10DET._file_paths_reader(dataset_path)

    def download(self, force: bool = False) -> bool:
        """
        Downloads the GC10-DET dataset.

        - `force: bool = False` Specifies whether the dataset should also be downloded if the
            destination directory is not empty. If True, the data in the destination directory are
            deleted before the download.
        """
        return GC10DET._downloader.download(target=self.root,
                                            dataset_name=self.base_dir,
                                            force=force)
