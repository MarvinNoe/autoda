import logging

from .config.yaml_loader import load_config, load_plugins

_LOG = logging.getLogger(__name__)

PLUGINS_KEY = 'plugins'
PATHS_KEY = 'paths'
NAMES_KEY = 'names'

EXPERIMENT_KEY = 'experiment'

ROUTINE_KEY = 'routine'
DATASET_KEY = 'dataset'
MODEL_KEY = 'model'


class RoutineExecutor():
    def __init__(self, *, name: str, config_file: str):
        self.name = name
        self.config_file = config_file

    def exec(self) -> None:
        """
        Executes the experiment.
        """
        _LOG.info('Executing experiment \'%s\' ...', self.name)

        load_plugins(self.config_file)

        config = load_config(self.config_file)
        routine_config = config.pop('routine', None)

        if routine_config is None:
            raise ValueError(f'No routine configuration defined in {self.config_file}')

        routine_config.create_instance().exec(self.config_file)

        _LOG.info('... experiment %s finished!', self.name)
