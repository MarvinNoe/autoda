from typing import Dict

from autoda.collections import AutoDaColleciton

my_collection_name = 'example_collection'
my_collection = AutoDaColleciton(my_collection_name)


class MyObj:
    def __init__(self) -> None:
        print("MyObj!")


my_collection.register(MyObj.__name__, MyObj())


def name() -> str:
    """
    Returns the name of the factory plugin.
    """
    return __name__


def version() -> str:
    """
    Returns the version of the factory plugin.
    """
    return "0.1.0.dev0"


def collections() -> Dict[str, AutoDaColleciton]:
    return {
        my_collection_name: my_collection
    }
