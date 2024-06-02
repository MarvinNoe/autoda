from typing import Dict

import torch
from torchvision.models.detection.ssd import SSD300_VGG16_Weights
from torchvision.transforms import v2

from autoda.collections import AutoDaCollection

from .globals import (
    TRANSF_CREATOR_COLL_NAME,
    TORCH_DATA_TRANSF_COLL_NAME,
    WEIGHTS_COLL_NAME,
    SA_COLOR_TRANSF_COLL_NAME,
    SA_GEOMETRY_TRANSF_COLL_NAME
)
from ..fastaa import op_creator as fastaa
from ..trivial_augment import op_creators as ta
from ..smart_augment import op_creators as sa


# PyTroch dataset transforms collection
_TORCH_DATA_TRANSF_COLL = AutoDaCollection(TORCH_DATA_TRANSF_COLL_NAME)

object_detection_transform_v2 = v2.Compose([
    v2.ToImage(),
    v2.ToDtype(torch.float32, scale=True)
])

_TORCH_DATA_TRANSF_COLL.register(
    f'{str(SSD300_VGG16_Weights.COCO_V1)}',
    object_detection_transform_v2
)


# Weights collection
_WEIGHTS_COLL = AutoDaCollection(WEIGHTS_COLL_NAME)
_WEIGHTS_COLL.register(str(SSD300_VGG16_Weights.COCO_V1), SSD300_VGG16_Weights.COCO_V1)


# Traonsforms creator collection
_TRANSF_CREATOR_COLL = AutoDaCollection(TRANSF_CREATOR_COLL_NAME)
_TRANSF_CREATOR_COLL.register('fastaa.default_transforms', fastaa.default_transform_creators)
_TRANSF_CREATOR_COLL.register('fastaa.gc10_transforms', fastaa.gc10_transform_creators)
_TRANSF_CREATOR_COLL.register('trivial_aug.default_transforms', ta.default_transform_creators)
# use gc10 transforms definition of fastaa for trival augment
_TRANSF_CREATOR_COLL.register('trivial_aug.gc10_transforms', fastaa.gc10_transform_creators)


# SmartAugment transforms creator collection
_SA_COLOR_TRANSF_COLL = AutoDaCollection(SA_COLOR_TRANSF_COLL_NAME)
_SA_COLOR_TRANSF_COLL.register(
    'smart_aug.default_color_transforms',
    sa.default_col_transform_creators
)

_SA_COLOR_TRANSF_COLL.register(
    'smart_aug.gc10_color_transforms',
    sa.gc10_col_transform_creators
)

_SA_GEOMETRY_TRANSF_COLL = AutoDaCollection(SA_GEOMETRY_TRANSF_COLL_NAME)
_SA_GEOMETRY_TRANSF_COLL.register(
    'smart_aug.default_geometry_transforms',
    sa.default_geo_transform_creators
)

_SA_GEOMETRY_TRANSF_COLL.register(
    'smart_aug.gc10_geometry_transforms',
    sa.gc10_geo_transform_creators
)


def name() -> str:
    """
    Returns the name of the collection plugin.
    """
    return "Collection Plugin"


def version() -> str:
    """
    Returns the version of the collection plugin.
    """
    return "0.1.0.dev0"


def collections() -> Dict[str, AutoDaCollection]:
    """
    Returns the collections of the collection plugin.
    """
    return {
        TRANSF_CREATOR_COLL_NAME: _TRANSF_CREATOR_COLL,
        TORCH_DATA_TRANSF_COLL_NAME: _TORCH_DATA_TRANSF_COLL,
        WEIGHTS_COLL_NAME: _WEIGHTS_COLL,
        SA_COLOR_TRANSF_COLL_NAME: _SA_COLOR_TRANSF_COLL,
        SA_GEOMETRY_TRANSF_COLL_NAME: _SA_GEOMETRY_TRANSF_COLL
    }
