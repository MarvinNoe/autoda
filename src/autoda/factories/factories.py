from .factory import AutoDaFactory
from ..collections import AutoDaCollection


# built-in factory collection
_factory_collection = AutoDaCollection('factory_collection')
""" The collection of factory instances """


def register_factory(name: str, factory: AutoDaFactory) -> None:
    """
    Registers the specified factory within the factory collection.
    """
    _factory_collection.register(name, factory)


def get_factory(name: str) -> AutoDaFactory:
    """
    Returns the factory with the specified name.
    """
    return _factory_collection.get(name)
