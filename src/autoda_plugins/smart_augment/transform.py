import logging
import random

from typing import Any, List, Tuple, Callable

from torchvision.transforms import v2

_LOG = logging.getLogger(__name__)


class SmartAugmentTransform(v2.Transform):
    def __init__(
        self,
        col_transform_creators: List[Tuple[Callable[[float], v2.Transform], float, float]],
        geo_transforms_creators: List[Tuple[Callable[[float], v2.Transform], float, float]],
        level_col: float,
        level_geo: float,
        num_col_trans: int,
        num_geo_trans: int,
        prob: float
    ):
        if num_col_trans > len(col_transform_creators):
            raise ValueError(
                f'num_col_trans ({num_col_trans}) must be smaller or equal to len(geo_transforms) '
                f'({len(col_transform_creators)})'
            )

        if num_geo_trans > len(geo_transforms_creators):
            raise ValueError(
                f'num_geo_trans ({num_geo_trans}) must be smaller or equal to len(geo_transforms) '
                f'({len(geo_transforms_creators)})'
            )

        if level_col > 1.0:
            raise ValueError(
                f'level_col ({level_col}) must be smaller or equal to 1.0'
            )

        if level_geo > 1.0:
            raise ValueError(
                f'level_geo ({level_geo}) must be smaller or equal to 1.0'
            )

        if prob > 1.0:
            raise ValueError(
                f'prob ({prob}) must be smaller or equal to 1.0'
            )

        super().__init__()

        self.col_transform_creators = col_transform_creators
        self.geo_transforms_creators = geo_transforms_creators
        self.level_col = level_col
        self.level_geo = level_geo
        self.num_col_trans = num_col_trans
        self.num_geo_trans = num_geo_trans
        self.prob = prob

    def forward(self, *inputs: Any) -> Any:
        # check if transform should be applied
        if self.prob < random.random():
            return inputs

        col_trans_creators = random.sample(self.col_transform_creators, self.num_col_trans)
        geo_trans_creators = random.sample(self.geo_transforms_creators, self.num_geo_trans)

        col_transforms = self._create_v2_transforms(col_trans_creators, self.level_col)
        geo_transforms = self._create_v2_transforms(geo_trans_creators, self.level_geo)

        transform = v2.Compose(col_transforms + geo_transforms)

        return transform(inputs)

    def _create_v2_transforms(
        self,
        transforms_creators: List[Tuple[Callable[[float], v2.Transform], float, float]],
        level: float
    ) -> List[v2.Transform]:

        transforms: List[v2.Transform] = []

        for creator, min_mag, max_mag in transforms_creators:
            mag = level * (max_mag - min_mag) + min_mag
            transforms.append(creator(mag))

        return transforms
