from typing import Dict, Protocol, Type
from typing_extensions import runtime_checkable

from ..abstract import Routine


@runtime_checkable
class RoutineInterface(Protocol):
    """
    A routine module implements a function called routines.
    This function returns a dictionary mapping routine names to corresponding routine constructor.
    In this way, routine constructors can be added to the routine factory dynamically.
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
    def routines() -> Dict[str, Type[Routine]]:
        """
        Returns a dictionary mapping routine names to corresponding routine constructor.
        The returnd dictionary is added to the routine factory.
        """
