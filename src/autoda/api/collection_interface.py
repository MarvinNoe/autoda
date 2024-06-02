from typing import Dict, Protocol
from typing_extensions import runtime_checkable

from ..collections import AutoDaCollection


@runtime_checkable
class CollectionInterface(Protocol):
    """
    A collection module implements a function called collections.
    This function returns a dictionary mapping collection names to corresponding collection
    instances. In this way, collections can be added to the collection of collections dynamically.
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
    def collections() -> Dict[str, AutoDaCollection]:
        """
        Returns a dictionary that maps collection names to the corresponding collection instances.
        The returned key-value pairs are added to the collection of collections.
        """
