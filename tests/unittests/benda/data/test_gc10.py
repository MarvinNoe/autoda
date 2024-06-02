from PIL import Image
from pathlib import Path

import torch

from helpers.files import TmpFile

from autoda_plugins.datasets.torch.vision_dataset import AutoDaVisionDataset
import autoda_plugins.datasets.torch.gc10 as gc10

FILE_PATH = Path(__file__).parent.resolve()
GC10_TEST_PATH = (FILE_PATH / '..' / '..' / '..' / 'data' / 'gc10det').resolve()
TMP_PATH = (FILE_PATH / '..' / '..' / '..' / 'tmp').resolve()

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

EXAMPLE_XML = '''<?xml version="1.0"?>
<annotation>
    <object>
        <name>3_yueyawan</name>
        <bndbox>
            <xmin>10</xmin>
            <ymin>15</ymin>
            <xmax>20</xmax>
            <ymax>25</ymax>
        </bndbox>
    </object>
    <object>
        <name>9_zhehen</name>
        <bndbox>
            <xmin>100</xmin>
            <ymin>150</ymin>
            <xmax>200</xmax>
            <ymax>250</ymax>
        </bndbox>
    </object>
</annotation>'''

INVALID_TARGET_EXAMPLE_XML = '''<?xml version="1.0"?>
<annotation>
    <object>
        <name>3_yueyawan</name>
        <bndbox>
            <xmax>20</xmax>
            <ymax>25</ymax>
        </bndbox>
    </object>
    <object>
        <name>9_zhehen</name>
        <bndbox>
            <xmin>100</xmin>
            <ymin>150</ymin>
            <xmax>200</xmax>
            <ymax>250</ymax>
        </bndbox>
    </object>
</annotation>'''

MISSING_TARGET_EXAMPLE_XML = '''<?xml version="1.0"?>
<annotation>
    <object>
        <name>3_yueyawan</name>
    </object>
    <object>
        <name>9_zhehen</name>
        <bndbox>
            <xmin>100</xmin>
            <ymin>150</ymin>
            <xmax>200</xmax>
            <ymax>250</ymax>
        </bndbox>
    </object>
</annotation>'''

INVALID_LABEL_EXAMPLE_XML = '''<?xml version="1.0"?>
<annotation>
    <object>
        <name>asdf</name>
        <bndbox>
            <xmin>10</xmin>
            <ymin>15</ymin>
            <xmax>20</xmax>
            <ymax>25</ymax>
        </bndbox>
    </object>
    <object>
        <name>9_zhehen</name>
        <bndbox>
            <xmin>100</xmin>
            <ymin>150</ymin>
            <xmax>200</xmax>
            <ymax>250</ymax>
        </bndbox>
    </object>
</annotation>'''

MISSING_LABEL_EXAMPLE_XML = '''<?xml version="1.0"?>
<annotation>
    <object>
        <bndbox>
            <xmin>10</xmin>
            <ymin>15</ymin>
            <xmax>20</xmax>
            <ymax>25</ymax>
        </bndbox>
    </object>
    <object>
        <name>9_zhehen</name>
        <bndbox>
            <xmin>100</xmin>
            <ymin>150</ymin>
            <xmax>200</xmax>
            <ymax>250</ymax>
        </bndbox>
    </object>
</annotation>'''

DATASET_LABLE_TYPO_XML = '''<?xml version="1.0"?>
<annotation>
    <object>
        <name>10_yaozhe</name>
        <bndbox>
            <xmin>10</xmin>
            <ymin>15</ymin>
            <xmax>20</xmax>
            <ymax>25</ymax>
        </bndbox>
    </object>
    <object>
        <name>10_yaozhed</name>
        <bndbox>
            <xmin>100</xmin>
            <ymin>150</ymin>
            <xmax>200</xmax>
            <ymax>250</ymax>
        </bndbox>
    </object>
</annotation>'''


def read_example_target(xml_content: str) -> None:
    xml_file = TmpFile(TMP_PATH / 'example_target.xml', xml_content)
    result = gc10._gc10det_v1_load_targets(
        xml_file.file_path, list(gc10.GC10DET_LABEL_MAP.keys()))

    boxes = result['boxes'].tolist()
    labels = result['labels'].tolist()

    return boxes, labels


def test_gc10det_label_map() -> None:
    assert set(gc10.GC10DET_LABEL_MAP.keys()) == set(GC10DET_LABEL_MAP.keys())
    for key in GC10DET_LABEL_MAP.keys():
        assert gc10.GC10DET_LABEL_MAP[key] == GC10DET_LABEL_MAP[key]


def test_gc10det_v1_load_paths_len() -> None:
    img_paths, target_paths = gc10._gc10det_v1_load_paths(str(GC10_TEST_PATH))
    file_count = len(list(GC10_TEST_PATH.rglob('*.jpg')))
    assert len(img_paths) == file_count
    assert len(img_paths) == len(target_paths)


def test_gc10det_v1_load_paths_order() -> None:
    img_paths, target_paths = gc10._gc10det_v1_load_paths(str(GC10_TEST_PATH))
    assert Path(img_paths[0]).stem == Path(target_paths[0]).stem
    assert Path(img_paths[1]).stem == Path(target_paths[1]).stem


def test_gc10det_v1_load_paths_relative_to_dataset_path() -> None:
    img_paths, target_paths = gc10._gc10det_v1_load_paths(str(GC10_TEST_PATH))
    first_img_path = GC10_TEST_PATH / img_paths[0]
    first_target_path = GC10_TEST_PATH / target_paths[0]

    assert first_img_path.exists()
    assert first_target_path.exists()


def test_gc10det_v1_load_targets_key_count() -> None:
    xml_file = TmpFile(TMP_PATH / 'example_target.xml', EXAMPLE_XML)
    result = gc10._gc10det_v1_load_targets(
        xml_file.file_path, list(gc10.GC10DET_LABEL_MAP.keys()))
    assert len(result.keys()) == 2


def test_gc10det_v1_load_targets_keys() -> None:
    xml_file = TmpFile(TMP_PATH / 'example_target.xml', EXAMPLE_XML)
    result = gc10._gc10det_v1_load_targets(
        xml_file.file_path, list(gc10.GC10DET_LABEL_MAP.keys()))
    assert 'boxes' in result.keys()
    assert 'labels' in result.keys()


def test_gc10det_v1_load_targets_valid_xml() -> None:
    xml_file = TmpFile(TMP_PATH / 'example_target.xml', EXAMPLE_XML)
    result = gc10._gc10det_v1_load_targets(
        xml_file.file_path, list(gc10.GC10DET_LABEL_MAP.keys()))

    boxes = result['boxes']
    assert isinstance(boxes, torch.Tensor) is True

    lists = boxes.tolist()

    assert len(lists) == 2
    assert len(lists[0]) == 4
    assert len(lists[1]) == 4
    assert lists[0][0] == 10
    assert lists[0][1] == 15
    assert lists[0][2] == 20
    assert lists[0][3] == 25
    assert lists[1][0] == 100
    assert lists[1][1] == 150
    assert lists[1][2] == 200
    assert lists[1][3] == 250

    labels = result['labels']
    assert isinstance(labels, torch.Tensor)

    lists = labels.tolist()
    keys = list(GC10DET_LABEL_MAP.keys())

    assert len(lists) == 2
    assert lists[0] == keys.index('3_yueyawan')
    assert lists[1] == keys.index('9_zhehen')


def test_gc10det_v1_load_targets_labels_invalid_target() -> None:
    boxes, labels = read_example_target(INVALID_TARGET_EXAMPLE_XML)
    keys = list(GC10DET_LABEL_MAP.keys())

    assert len(labels) == 1
    assert labels[0] == keys.index('9_zhehen')

    assert len(boxes) == 1
    assert boxes[0][0] == 100
    assert boxes[0][1] == 150
    assert boxes[0][2] == 200
    assert boxes[0][3] == 250


def test_gc10det_v1_load_targets_labels_missing_target() -> None:
    boxes, labels = read_example_target(MISSING_TARGET_EXAMPLE_XML)
    keys = list(GC10DET_LABEL_MAP.keys())

    assert len(labels) == 1
    assert labels[0] == keys.index('9_zhehen')

    assert len(boxes) == 1
    assert boxes[0][0] == 100
    assert boxes[0][1] == 150
    assert boxes[0][2] == 200
    assert boxes[0][3] == 250


def test_gc10det_v1_load_targets_invalid_labels() -> None:
    boxes, labels = read_example_target(INVALID_LABEL_EXAMPLE_XML)
    keys = list(GC10DET_LABEL_MAP.keys())

    assert len(labels) == 1
    assert labels[0] == keys.index('9_zhehen')

    assert len(boxes) == 1
    assert boxes[0][0] == 100
    assert boxes[0][1] == 150
    assert boxes[0][2] == 200
    assert boxes[0][3] == 250


def test_gc10det_v1_load_targets_missing_labels() -> None:
    boxes, labels = read_example_target(MISSING_LABEL_EXAMPLE_XML)
    keys = list(GC10DET_LABEL_MAP.keys())

    assert len(labels) == 1
    assert labels[0] == keys.index('9_zhehen')

    assert len(boxes) == 1
    assert boxes[0][0] == 100
    assert boxes[0][1] == 150
    assert boxes[0][2] == 200
    assert boxes[0][3] == 250


def test_gc10det_v1_load_last_label_typo() -> None:
    _, labels = read_example_target(DATASET_LABLE_TYPO_XML)
    keys = list(GC10DET_LABEL_MAP.keys())

    assert labels[0] == keys.index('10_yaozhed')
    assert labels[1] == keys.index('10_yaozhed')


def test_gc10det_base_class() -> None:
    assert issubclass(gc10.GC10DET, AutoDaVisionDataset)


def test_gc10det_load_image() -> None:
    gc10det = gc10.GC10DET(GC10_TEST_PATH.parent)
    assert isinstance(gc10det.load_image(0), Image.Image)


def test_gc10det_load_target() -> None:
    gc10det = gc10.GC10DET(GC10_TEST_PATH.parent)
    target = gc10det.load_target(0)

    assert 'boxes' in target.keys()
    assert 'labels' in target.keys()
    assert 'image_id' in target.keys()
    assert target['image_id'].item() == 0


def test_gc10det_file_pathss_len() -> None:
    img_paths, target_paths = gc10.GC10DET(GC10_TEST_PATH.parent).file_paths(GC10_TEST_PATH)
    file_count = len(list(GC10_TEST_PATH.rglob('*.jpg')))
    assert len(img_paths) == file_count
    assert len(img_paths) == len(target_paths)


def test_gc10det_file_paths_order() -> None:
    img_paths, target_paths = gc10.GC10DET(GC10_TEST_PATH.parent).file_paths(GC10_TEST_PATH)
    assert Path(img_paths[0]).stem == Path(target_paths[0]).stem
    assert Path(img_paths[1]).stem == Path(target_paths[1]).stem


def test_gc10det_file_paths_relative_to_dataset_path() -> None:
    img_paths, target_paths = gc10.GC10DET(GC10_TEST_PATH.parent).file_paths(GC10_TEST_PATH)
    first_img_path = GC10_TEST_PATH / img_paths[0]
    first_target_path = GC10_TEST_PATH / target_paths[0]

    assert first_img_path.exists()
    assert first_target_path.exists()
