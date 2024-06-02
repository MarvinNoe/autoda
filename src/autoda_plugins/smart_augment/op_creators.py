import logging

from typing import List, Tuple, Callable

from torchvision.transforms import v2

from ..utils.torch.transforms import creators as c

_LOG = logging.getLogger(__name__)


def default_col_transform_creators() -> List[Tuple[Callable[[float], v2.Transform], float, float]]:
    return [
        (c.auto_contrast, 0, 1),  # 0
        (c.equalize, 0, 1),  # 1
        (c.solarize, 0, 1),  # 2
        # solarize max_mag changed form 256 to 1, because the pixel values in this
        # implementation are between 0 and 1
        (c.contrast, 0.1, 1.9),  # 3
        (c.color, 0.1, 1.9),  # 4
        (c.brightness, 0.1, 1.9),  # 5
        (c.sharpness, 0.1, 1.9),  # 6
    ]


def default_geo_transform_creators() -> List[Tuple[Callable[[float], v2.Transform], float, float]]:
    return [
        (c.rotate, -30, 30),  # 0
        (c.shear_x, -0.3, 0.3),  # 1
        (c.shear_y, -0.3, 0.3),  # 2
        (c.translate_x, -0.45, 0.45),  # 3
        (c.translate_y, -0.45, 0.45),  # 4
    ]


def gc10_col_transform_creators() -> List[Tuple[Callable[[float], v2.Transform], float, float]]:
    return [
        (c.solarize, 0, 1),  # 0
        # solarize max_mag changed form 256 to 1, because the pixel values in this
        # implementation are between 0 and 1
        (c.posterize, 4, 8),  # 1
        (c.color, 0.1, 1.9),  # 2
        (c.sharpness, 0.1, 1.9),  # 3
    ]


def gc10_geo_transform_creators() -> List[Tuple[Callable[[float], v2.Transform], float, float]]:
    return [
        (c.shear_x, -0.3, 0.3),  # 0
        (c.shear_y, -0.3, 0.3),  # 1
        (c.translate_x, -0.45, 0.45),  # 2
        (c.translate_y, -0.45, 0.45),  # 3
    ]
