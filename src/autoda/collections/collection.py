from typing import Dict, Optional, List, Any


class AutoDaCollection:

    def __init__(self, name: str) -> None:
        """
        Initializes the collection.
        """
        self.name = name
        self._collection: Dict[str, Any] = {}

    def register(self, key: str, value: Any) -> None:
        """
        Registers the specified key value pair within the collection.

        `key: str` The key that identifies the value.

        `value: T` A value of type T.
        """
        if key in self._collection:
            raise KeyError(f'{self.name}: key \'{key}\' already exists in {self.name}!') from None
        self._collection[key] = value

    def unregister(self, key: str) -> Optional[Any]:
        """
        Unregisters the value identified by the specified key from the collections dict.

        `key: str` The key that identifies the value to be unregistered.

        Retruns the unregistered value.
        """
        return self._collection.pop(key, None)

    def get(self, key: str) -> Any:
        """
        Returns the value identified by the given key.
        """
        if key not in self._collection:
            raise ValueError(f'{self.name}: unknown type {key}!') from None
        return self._collection[key]

    def registered_keys(self) -> List[str]:
        """
        Returns the list of the registered keys.
        """
        return [*self._collection]
