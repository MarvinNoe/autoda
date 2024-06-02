from typing import Dict, Callable, Any


class MyDataset:
    def __init__(self) -> None:
        print("MyDataset")


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


def datasets() -> Dict[str, Callable[..., Any]]:
    return {
        MyDataset.__name__: MyDataset
    }
