from typing import Dict

from ..globals import DATASET_FACTORY_NAME
from ..factories import AutoDaFactory

dataset_factory = AutoDaFactory(DATASET_FACTORY_NAME)
""" Dataset factory. """


def name() -> str:
    """
    Returns the name of the loadable component.
    """
    return 'Dataset Factory Plugin'


def version() -> str:
    """
    Returns the version of the loadable component.
    """
    return '0.1.0'


def factories() -> Dict[str, AutoDaFactory]:
    """
    Returns a dictionary mapping model names to corresponding modedl constructor.
    The returnd dictionary is added to the model factory.
    """
    return {
        DATASET_FACTORY_NAME: dataset_factory
    }
