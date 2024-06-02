from typing import Dict

from autoda.factories import AutoDaFactory

my_factory_name = 'example_factory'
my_factory = AutoDaFactory(my_factory_name)


class MyObj:
    def __init__(self) -> None:
        print("MyObj!")


my_factory.register(MyObj.__name__, MyObj)


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


def factories() -> Dict[str, AutoDaFactory]:
    return {
        my_factory_name: my_factory
    }
