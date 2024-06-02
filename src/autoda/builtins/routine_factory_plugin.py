from typing import Dict

from ..globals import ROUTINE_FACTORY_NAME
from ..factories import AutoDaFactory


routine_factory = AutoDaFactory(ROUTINE_FACTORY_NAME)
""" Routine factory. """


def name() -> str:
    """
    Returns the name of the loadable component.
    """
    return 'Training Factory Plugin'


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
        ROUTINE_FACTORY_NAME: routine_factory
    }
