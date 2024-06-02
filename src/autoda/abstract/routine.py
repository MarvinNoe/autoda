from abc import ABC, abstractmethod


class Routine(ABC):
    @abstractmethod
    def exec(self, config_file: str) -> None:
        pass
