from typing import Callable, Dict, Optional, List, Any


class AutoDaFactory:
    def __init__(self, name: str = 'AutoDaFactory') -> None:
        """
        Initializes the factory.
        """
        self.name = name
        self._constructors: Dict[str, Callable[..., Any]] = {}

    def register(self, key: str, constructor: Callable[..., Any]) -> None:
        """
        Registers the specified constructor in the dict of constructors.

        `key: str` The key that identifies the constructor.

        `constructor: Callable[..., Any]` A callable object constructor of type T.
        """
        if key in self._constructors:
            raise KeyError(f'{self.name}: key \'{key}\' already exists in {self.name}!') from None
        self._constructors[key] = constructor

    def unregister(self, key: str) -> Optional[Callable[..., Any]]:
        """
        Unregisters the constructor identified by the specified key from the dict of
        constructors.

        `key: str` The key that identifies the constructor to be unregistered.

        Retruns the unregistered constructor.
        """
        return self._constructors.pop(key, None)

    def create(self, key: str, *args: Any, **kwargs: Any) -> Any:
        """
        Creates an instance using the constructor identified by the given key.

        `key: str` The key that identifies the constructor.

        `args: Any` Positional arguments the constructor needs for initialization.

        `kwargs: Any` Keyword arguments the constructor needs for initialization.

        Returnst the created instace of type T.
        """
        if key not in self._constructors:
            raise ValueError(f'{self.name}: unknown type {key}!') from None
        return self._constructors[key](*args, **kwargs)

    def registered_keys(self) -> List[str]:
        """
        Returns the list of the registered keys.
        """
        return [*self._constructors]
