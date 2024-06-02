import pytest
import pytest_mock.plugin as mock

from pathlib import Path
from PIL import Image
from typing import Dict, Any

import torch

from torchvision import tv_tensors

from autoda_plugins.datasets.torch.vision_dataset import AutoDaVisionDataset, FilePathsType

FILE_PATH = Path(__file__).parent.resolve()
GC10_TEST_PATH = (FILE_PATH / '..' / '..' / '..' / 'data' / 'gc10det').resolve()
TMP_PATH = (FILE_PATH / '..' / '..' / '..' / 'tmp').resolve()


class MyAutoDaVisionDataset(AutoDaVisionDataset):
    def __init__(self, root: str, base_dir: str):
        super().__init__(
            root,
            base_dir
        )

    def load_image(self, index: int) -> Image.Image:
        return Image.open(self.image_path(index))

    def load_target(self, index: int) -> Dict[str, Any]:
        return {
            'boxes': [[10, 15, 20, 25]],
            'labels': [1],
            'image_id': index,
            'taget_file': self.target_path(index),
            'image_file': self.image_path(index)
        }


class SingleTargetFileDataset(MyAutoDaVisionDataset):
    def __init__(self, root: str, base_dir: str):
        super().__init__(
            root,
            base_dir
        )

    def file_paths(self, dataset_path: str) -> FilePathsType:
        return (
            ['images/images/crease/img_01_425382900_00002.jpg',
             'images/images/crescent_gap/img_01_424799300_01133.jpg'],
            'label/label/img_01_425382900_00002.xml'
        )


class MultiTargetFileDataset(MyAutoDaVisionDataset):
    def __init__(self, root: str, base_dir: str):
        super().__init__(
            root,
            base_dir
        )

    def file_paths(self, dataset_path: str) -> FilePathsType:
        return (
            ['images/images/crease/img_01_425382900_00002.jpg',
             'images/images/crescent_gap/img_01_424799300_01133.jpg'],
            ['label/label/img_01_425382900_00002.xml',
             'label/label/img_01_424799300_01133.xml']
        )


def test_autodavisiondataset_abstract_class(mocker: mock.MockerFixture):
    with pytest.raises(NotImplementedError):
        AutoDaVisionDataset(GC10_TEST_PATH.parent, GC10_TEST_PATH.name)

    mocker.patch.object(AutoDaVisionDataset, 'file_paths', return_value=(
        ['images/images/crease/img_01_425382900_00002.jpg'],
        ['label/label/img_01_425382900_00002.xml'])
    )

    dataset = AutoDaVisionDataset(GC10_TEST_PATH.parent, GC10_TEST_PATH.name)

    with pytest.raises(NotImplementedError):
        dataset.load_image(0)

    with pytest.raises(NotImplementedError):
        dataset.load_target(0)


def test_autodavisiondataset_properties():
    dataset = MultiTargetFileDataset(GC10_TEST_PATH.parent, GC10_TEST_PATH.name)

    assert dataset.use_transforms_v2 is True
    dataset.use_transforms_v2 = False
    assert dataset.use_transforms_v2 is False

    assert dataset.base_dir == str(GC10_TEST_PATH.name)


def test_autodavisiondataset_getitem():
    dataset = MultiTargetFileDataset(GC10_TEST_PATH.parent, GC10_TEST_PATH.name)

    dataset.use_transforms_v2 = False

    img, target = dataset[0]

    assert isinstance(img, Image.Image)
    assert isinstance(target, dict)

    assert target['boxes'] == [[10, 15, 20, 25]]
    assert target['labels'] == [1]
    assert target['image_id'] == 0


def test_autodavisiondataset_getitem_v2_convert():
    dataset = MultiTargetFileDataset(GC10_TEST_PATH.parent, GC10_TEST_PATH.name)

    dataset.use_transforms_v2 = True

    img, target = dataset[0]

    assert isinstance(img, torch.Tensor)
    assert isinstance(target, dict)

    boxes = target['boxes']
    expected_boxes = tv_tensors.BoundingBoxes(
        [[10, 15, 20, 25]],
        format=tv_tensors.BoundingBoxFormat.XYXY,
        canvas_size=tv_tensors.Image(img).shape[-2:]
    )

    assert isinstance(boxes, tv_tensors.BoundingBoxes)
    assert len(boxes) == len(expected_boxes)
    assert torch.equal(boxes[0], expected_boxes[0])
    assert boxes.format == expected_boxes.format
    assert boxes.canvas_size == expected_boxes.canvas_size

    assert target['labels'] == [1]
    assert target['image_id'] == 0


def test_autodavisiondataset_single_multiple_targets():
    single_dataset = SingleTargetFileDataset(GC10_TEST_PATH.parent, GC10_TEST_PATH.name)
    multiple_dataset = MultiTargetFileDataset(GC10_TEST_PATH.parent, GC10_TEST_PATH.name)

    assert len(single_dataset) == 2
    assert len(multiple_dataset) == 2

    _, first = single_dataset[0]
    _, sec = single_dataset[1]

    assert first['taget_file'] == sec['taget_file']
    assert first['image_file'] != sec['image_file']

    _, first = multiple_dataset[0]
    _, sec = multiple_dataset[1]

    assert first['taget_file'] != sec['taget_file']
    assert first['image_file'] != sec['image_file']
