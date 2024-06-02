from typing import Protocol
from typing_extensions import runtime_checkable


@runtime_checkable
class AutoDaInterface(Protocol):
    """
    The AutoDaInterface defines the interface of the AutoDa plugins.

    To implemente a plugin, create python module and define following functions.
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
