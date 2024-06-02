import logging
import random

from typing import Any, List, Tuple, Callable, Optional

from torchvision.transforms import v2

_LOG = logging.getLogger(__name__)


class TrivialAugment(v2.Transform):
    def __init__(self, transforms: List[Tuple[Optional[Callable], float, float]]):
        super().__init__()

        self.transforms = transforms

    def forward(self, *inputs: Any) -> Any:
        # sample transform
        transform_creator, min_mag, max_mag = random.choice(self.transforms)

        # identity
        if transform_creator is None:
            return inputs

        # sample level and calculate magnitude
        level = float(random.randint(0, 30)) / 30.0
        mag = level * (max_mag - min_mag) + min_mag

        # create transform
        transform = transform_creator(mag)

        return transform(inputs)
