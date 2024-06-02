import copy
import yaml

from typing import Dict, Any

from .config_object import ConfigObject
from ..plugins_loader import PluginsLoader


def load_plugins_dict(plugins_dict: Dict[str, Any]):
    try:
        PluginsLoader.from_json(plugins_dict).load_plugins()
    except KeyError as e:
        raise KeyError('An error occurred while loading the plugins: {}'.format(e)) from e


def config_objects(config_dict: Dict[str, Any]) -> Dict[str, ConfigObject]:
    config_objects = {}
    config_dict_copy = copy.deepcopy(config_dict)

    for source, args in config_dict_copy.items():
        type_key = args.pop('type', None)

        if type_key is None:
            raise ValueError(
                f'The configuration of \'{source}\' expects a type.'
            )

        config_objects[source] = ConfigObject(source, type_key, args if args else None)

    return config_objects


def load_plugins(config_file):
    with open(config_file) as f:
        config_dict = yaml.safe_load(f)

    plugins_dict = config_dict.pop('plugins', None)

    if plugins_dict is None:
        raise KeyError('No plugins are defined in {}'.format(config_file))

    load_plugins_dict(plugins_dict)


def load_config(config_file: str) -> Dict[str, ConfigObject]:
    with open(config_file) as f:
        config_dict = yaml.safe_load(f)

    config_dict.pop('plugins', None)

    return config_objects(config_dict)
