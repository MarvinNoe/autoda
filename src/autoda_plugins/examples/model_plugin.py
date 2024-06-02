from typing import Dict, Callable, Any


class MyModel:
    def __init__(self) -> None:
        print("MyModel")


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


def models() -> Dict[str, Callable[..., Any]]:
    return {
        MyModel.__name__: MyModel
    }
