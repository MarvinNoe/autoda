import random

from typing import Optional

import torch
import numpy as np


def set_random_seed(random_seed: Optional[int]) -> None:
    if random_seed is None:
        return
    random.seed(random_seed)
    torch.manual_seed(random_seed)
    np.random.seed(random_seed)
