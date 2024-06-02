from typing import Protocol, Any


class AutoDaLoader(Protocol):
    def load(self, module: Any) -> None:
        """
        Loads the given module.
        """
