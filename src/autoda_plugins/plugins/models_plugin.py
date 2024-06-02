from typing import Dict, Callable, Any

from ..models.torch.ssd_vgg16 import create_ssd_vgg16


def name() -> str:
    """
    Returns the name of the dataset plugin.
    """
    return __name__


def version() -> str:
    """
    Returns the version of the dataset plugin.
    """
    return "0.1.0.dev0"


def models() -> Dict[str, Callable[..., Any]]:
    return {
        'torch.ssd_vgg16': create_ssd_vgg16
    }
