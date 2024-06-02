from torchvision.models.detection import _utils as _utils_detection
from torchvision.models.detection import ssd as torch_ssd

from helpers.vars import COMPLEX_TESTS

import autoda_plugins.models.torch.ssd_vgg16 as ssd


def test_create_ssd_vgg16_architecture() -> None:
    if not COMPLEX_TESTS:
        return

    size = 512
    num_classes = 11
    model = ssd.create_ssd_vgg16(size=size, num_classes=num_classes)

    in_channels = _utils_detection.retrieve_out_channels(model.backbone, (size, size))
    num_anchors = model.anchor_generator.num_anchors_per_location()

    assert isinstance(model, torch_ssd.SSD)
    assert isinstance(model.backbone, torch_ssd.SSDFeatureExtractorVGG)
    assert len(model.backbone.extra) == 5  # check if backone is an vgg16 model

    assert model.head.classification_head.num_columns == num_classes

    assert model.head.classification_head.module_list[0].in_channels == in_channels[0]
    assert model.head.classification_head.module_list[1].in_channels == in_channels[1]
    assert model.head.classification_head.module_list[2].in_channels == in_channels[2]
    assert model.head.classification_head.module_list[3].in_channels == in_channels[3]
    assert model.head.classification_head.module_list[4].in_channels == in_channels[4]
    assert model.head.classification_head.module_list[5].in_channels == in_channels[5]

    assert model.head.classification_head.module_list[0].out_channels == \
        num_classes * num_anchors[0]
    assert model.head.classification_head.module_list[1].out_channels == \
        num_classes * num_anchors[1]
    assert model.head.classification_head.module_list[2].out_channels == \
        num_classes * num_anchors[2]
    assert model.head.classification_head.module_list[3].out_channels == \
        num_classes * num_anchors[3]
    assert model.head.classification_head.module_list[4].out_channels == \
        num_classes * num_anchors[4]
    assert model.head.classification_head.module_list[5].out_channels == \
        num_classes * num_anchors[5]

    assert model.transform.min_size == (size,)
    assert model.transform.max_size == size


def test_create_ssd_vgg16_weights() -> None:
    if not COMPLEX_TESTS:
        return

    size = 300
    num_classes = 11

    # downloadig the weights kills the ci/cd docker container
    model = ssd.create_ssd_vgg16(
        size=size,
        num_classes=num_classes,
        weights=torch_ssd.SSD300_VGG16_Weights.COCO_V1
    )

    assert next(model.parameters())[0] != next(model.parameters())[1]

    model = ssd.create_ssd_vgg16(
        size=size,
        num_classes=num_classes
    )

    assert next(model.parameters())[0] == next(model.parameters())[1]
