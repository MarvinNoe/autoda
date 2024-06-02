from typing import Dict, Any

from hyperopt import hp

from .globals import POLICY_PREFIX, PROB_PREFIX, LEVEL_PREFIX


def fastaa_search_space_hyperopt(
    num_policies: int,
    num_ops: int,
    num_transforms: int,
    **kwargs
) -> Dict[str, Any]:
    space = {}
    for i in range(num_policies):
        for j in range(num_ops):
            space[f'{POLICY_PREFIX}_{i}_{j}'] = hp.choice(
                f'{POLICY_PREFIX}_{i}_{j}', list(range(0, num_transforms))
            )
            space[f'{PROB_PREFIX}_{i}_{j}'] = hp.uniform(f'{PROB_PREFIX}_{i}_{j}', 0.0, 1.0)
            space[f'{LEVEL_PREFIX}_{i}_{j}'] = hp.uniform(f'{LEVEL_PREFIX}_{i}_{j}', 0.0, 1.0)
    return space
