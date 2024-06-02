from typing import Dict, Any

from ray import tune

from .globals import (
    PROB,
    NUM_COLOR_TRANSFORMS,
    NUM_GEOMETRY_TRANSFORMS,
    LEVEL_COLOR,
    LEVEL_GEOMETRY
)


def smart_augment_default_search_space(
    total_color_transforms: int,
    total_geo_transforms: int,
    **kwargs
) -> Dict[str, Any]:
    return {
        PROB: tune.uniform(0.0, 1.0),
        NUM_COLOR_TRANSFORMS: tune.randint(1, total_color_transforms),
        NUM_GEOMETRY_TRANSFORMS: tune.randint(1, total_geo_transforms),
        LEVEL_COLOR: tune.uniform(0.0, 1.0),
        LEVEL_GEOMETRY: tune.uniform(0.0, 1.0)
    }
