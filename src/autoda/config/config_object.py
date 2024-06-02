import copy

from typing import Dict, Any, Optional

from ..factories import get_factory
from ..collections import get_collection


class ConfigObject:
    def __init__(self, source: str, type_key: str, args: Optional[Dict[str, Any]] = None):
        self.source = source
        self.type_key = type_key
        self.args = args

    def create_instance(self, **kwargs: Any) -> Any:
        if self.args is None:
            return get_collection(self.source).get(self.type_key)

        config_args = copy.deepcopy(self.args)

        for key, value in kwargs.items():
            if key in config_args:
                raise ValueError(
                    f"Key '{key}' exists in the configuration file and "
                    "is passed to create_instance()!"
                )
            config_args[key] = copy.deepcopy(value)

        for key, value in config_args.items():
            if isinstance(value, dict):
                type_key = value.pop('type', None)

                if type_key is None:
                    continue
                    # TODO: check if an error should be raised here
                    # raise ValueError(
                    #    f'The configuration of \'{key}\' expects a type.'
                    # )

                config_args[key] = ConfigObject(
                    source=key,
                    type_key=type_key,
                    args=value if value else None
                ).create_instance()

        return get_factory(self.source).create(self.type_key, **config_args)

    def get_arg(self, key: str) -> Any:
        """
        Returns the argument value identified by the given key.
        """
        if self.args is None or key not in self.args:
            raise KeyError(f'{self.source}: unknown type {key}!') from None

        args = copy.deepcopy(self.args)
        value = args[key]

        if not isinstance(value, dict):
            return value

        type_key = value.pop('type', None)

        if type_key is None:
            raise ValueError(
                f'The configuration of \'{key}\' expects a type.'
            )

        return ConfigObject(source=key, type_key=type_key, args=value if value else None)

    def set_arg(self, key: str, value: Any) -> None:
        """
        Sets the argument value identified by the given key.
        """
        if self.args is None:
            raise ValueError(f'{self.source} does not expect any arguments!')

        self.args[key] = value
