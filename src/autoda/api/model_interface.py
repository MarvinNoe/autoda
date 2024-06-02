from typing import Dict, Callable, Any, Type, Protocol
from typing_extensions import runtime_checkable


@runtime_checkable
class ModelInterface(Protocol):
    """
    A model module implements a function called models.
    This function returns a dictionary mapping model names to corresponding model constructor.
    In this way, model constructors can be added to the model factory dynamically.
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
    def models() -> Dict[str, Type[Callable[..., Any]]]:
        """
        Returns a dictionary mapping model names to corresponding modedl constructor.
        The returnd dictionary is added to the model factory.
        """
