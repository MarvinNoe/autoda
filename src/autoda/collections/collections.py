from .collection import AutoDaCollection

# built-in collections
_collections = AutoDaCollection('collections')
""" The collection of factory instances """


def register_collection(name: str, collection: AutoDaCollection) -> None:
    """
    Registers the specified collection within the collections.
    """
    _collections.register(name, collection)


def get_collection(name: str) -> AutoDaCollection:
    """
    Returns the factory with the specified name.
    """
    return _collections.get(name)
