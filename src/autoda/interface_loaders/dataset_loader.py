import logging

from ..api.dataset_interface import DatasetInterface
from ..globals import DATASET_FACTORY_NAME
from ..factories import get_factory


_LOG = logging.getLogger(__name__)


class DatasetLoader:
    def load(self, module: DatasetInterface) -> None:
        datasets = module.datasets()
        for name in datasets:
            get_factory(DATASET_FACTORY_NAME).register(name, datasets[name])
            _LOG.info('Dataset \'%s\' from %s (%s) added to dataset_factory!',
                      name, module.name(), module.version())
