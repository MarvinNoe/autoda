from typing import Dict, Protocol
from typing_extensions import runtime_checkable

from ..factories import AutoDaFactory


@runtime_checkable
class FactoryInterface(Protocol):
    """
    A factory module implements a function called factories.
    This function returns a dictionary mapping factory names to corresponding factory instances.
    In this way, factories can be added to the collection of factories dynamically.
    """
    @staticmethod
    def name() -> str:
        """
        Returns the name of the loadable component.
        """

    @staticmethod
    def version() -> str:
        """
        Returns the version of the loadable component.
        """

    @staticmethod
    def factories() -> Dict[str, AutoDaFactory]:
        """
        Returns a dictionary that maps factory names to the corresponding factory instances.
        The returned key-value pairs are added to the collection of factories.
        """
