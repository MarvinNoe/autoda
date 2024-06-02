from typing import Dict, Callable, Any, Type, Protocol
from typing_extensions import runtime_checkable


@runtime_checkable
class DatasetInterface(Protocol):
    """
    A dataset module implements a function called datasets.
    This function returns a dictionary mapping dataset names to corresponding dataset constructor.
    In this way, dataset constructors can be added to the dataset factory dynamically.
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
    def datasets() -> Dict[str, Type[Callable[..., Any]]]:
        """
        Returns a dictionary mapping dataset names to corresponding dataset constructor.
        The returnd dictionary is added to the dataset factory.
        """
