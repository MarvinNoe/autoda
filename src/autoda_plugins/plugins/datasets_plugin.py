from typing import Dict, Type

from ..datasets.torch.gc10 import GC10DET


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


def datasets() -> Dict[str, Type[GC10DET]]:
    return {
        'torch.GC10DET': GC10DET
    }
