from typing import Dict

from ..globals import MODEL_FACTORY_NAME
from ..factories import AutoDaFactory

model_factory = AutoDaFactory(MODEL_FACTORY_NAME)
""" Model factory. """


def name() -> str:
    """
    Returns the name of the loadable component.
    """
    return 'Model Factory Plugin'


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
        MODEL_FACTORY_NAME: model_factory
    }
