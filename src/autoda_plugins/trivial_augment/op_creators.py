import logging

from typing import Any, List, Tuple, Callable, Optional

from ..utils.torch.transforms import creators as c

_LOG = logging.getLogger(__name__)


def default_transform_creators() -> List[Tuple[Optional[Callable[..., Any]], float, float]]:
    return [
        (c.shear_x, -0.3, 0.3),  # 0
        (c.shear_y, -0.3, 0.3),  # 1
        (c.translate_x, -0.45, 0.45),  # 2
        (c.translate_y, -0.45, 0.45),  # 3
        (c.rotate, -30, 30),  # 4
        (c.auto_contrast, 0, 1),  # 5
        (c.invert, 0, 1),  # 6
        (c.equalize, 0, 1),  # 7
        (c.solarize, 0, 1),  # 8
        # solarize max_mag changed form 256 to 1, because the pixel values in this
        # implementation are between 0 and 1
        (c.posterize, 4, 8),  # 9
        (c.contrast, 0.1, 1.9),  # 10
        (c.color, 0.1, 1.9),  # 11
        (c.brightness, 0.1, 1.9),  # 12
        (c.sharpness, 0.1, 1.9),  # 13
        (c.cutout, 0, 0.05),  # 14
        # cutout max_mag changed from 0.2 to 0.5, because the value in this implementation
        # refers to the area and not to the side length.
        # (t.SamplePairing(imgs), 0, 0.4),  # 15
        (None, 0., 0.)  # 15 identity
    ]
