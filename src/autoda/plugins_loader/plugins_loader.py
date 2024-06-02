import importlib
import logging
import sys

from dataclasses import dataclass, field, Field
from types import ModuleType
from typing import List, Dict, Any, Type, Protocol, ClassVar

from ..interface_loader_map import interface_loader_map

_LOG = logging.getLogger(__name__)


class Loader(Protocol):
    __dataclass_fields__: ClassVar[dict[str, Field[Any]]]

    def load_plugins(self) -> None:
        """
        Loads the plugins.
        """


@dataclass
class PluginsLoader:
    paths: List[str] = field(default_factory=list)
    names: List[str] = field(default_factory=list)

    @classmethod
    def from_json(cls: Type[Loader], json_data: Dict[str, Any]) -> Loader:
        fields = cls.__dataclass_fields__.keys()
        attrs = {attr: json_data[attr] for attr in fields}
        return cls(**attrs)

    def load_plugins(self) -> None:
        if self.paths is not None:
            sys.path.extend(self.paths)

        for plugin_name in self.names:
            plugin = importlib.import_module(plugin_name)

            self.load(plugin)

    def load(self, plugin: ModuleType) -> None:
        is_plugin = False

        for interface, loader in interface_loader_map.items():
            # check if static methods defined by the interface are available in the plugin

            if isinstance(plugin, interface):  # type: ignore
                loader().load(plugin)
                is_plugin = True

        if not is_plugin:
            raise TypeError(
                f'Plugin {plugin.__name__} does not implement any interface!'
            )

        _LOG.info('Plugin %s (%s) successfully loaded!', plugin.__name__, plugin.version())
