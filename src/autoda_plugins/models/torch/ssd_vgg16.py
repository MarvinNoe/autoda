import logging

from typing import Optional, Any

from torchvision.models.detection import ssd
from torchvision.models.detection import _utils as _utils_detection

_LOG = logging.getLogger(__name__)
"""Module-level logger."""


def create_ssd_vgg16(
    *,
    size: int,
    num_classes: Optional[int] = 91,
    weights: Optional[ssd.SSD300_VGG16_Weights] = None,
    trainable_backbone_layers: Optional[int] = None,
    progress: bool = True,
    **kwargs: Any,
) -> ssd.SSD:
    """
    Creates an SSD model using a VGG16 backbone.

    - `size: int` Size of the input images. The input images are transformed by the model into
        square images of the specified size.

    - `num_classes: Optional[int]` Number of classes. Defaults to 91 (MS COCO classes).

    - `weights: Optional[ssd.SSD300_VGG16_Weights]` Pretrained weights. Defaults to ``None``.

    - `trainable_backbone_layers: Optional[int]` Number of trainable (not frozen) layers starting
        from final block. Valid values are between 0 and 5, with 5 meaning all backbone layers
        are trainable. If ``None`` is passed (the default) this value is set to 4.

    Retruns an SSD model of type ``torchvision.models.detection.ssd.SSD`` with the
    specified configuration.
    """
    model = ssd.ssd300_vgg16(
        weights=weights,
        progress=progress,
        trainable_backbone_layers=trainable_backbone_layers,
        **kwargs
    )

    in_channels = _utils_detection.retrieve_out_channels(model.backbone, (size, size))
    num_anchors = model.anchor_generator.num_anchors_per_location()

    model.head.classification_head = ssd.SSDClassificationHead(
        in_channels=in_channels,
        num_anchors=num_anchors,
        num_classes=num_classes,
    )

    model.transform.min_size = (size,)
    model.transform.max_size = size

    return model
