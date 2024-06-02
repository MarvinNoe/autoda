
import logging

from . import abstract
from . import api
from . import factories
from . import interface_loaders
from . import plugins_loader

from .experiment import RoutineExecutor


from .__version__ import __version__

__all__ = [
    'abstract',
    'api',
    'factories',
    'interface_loaders',
    'plugins_loader',
    'RoutineExecutor'
]


logging.getLogger(__name__).addHandler(logging.NullHandler())

plugins_loader.PluginsLoader(
    paths=[],
    names=[
        'autoda.builtins.model_factory_plugin',
        'autoda.builtins.dataset_factory_plugin',
        'autoda.builtins.routine_factory_plugin'
    ]
).load_plugins()
