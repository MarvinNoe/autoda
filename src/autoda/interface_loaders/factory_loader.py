import logging

from ..api import FactoryInterface
from ..factories import register_factory


_LOG = logging.getLogger(__name__)


class FactoryLoader:
    def load(self, module: FactoryInterface) -> None:
        factories = module.factories()
        for name in factories:
            register_factory(name, factories[name])
            _LOG.info('Factory \'%s\' from %s (%s) added to factory_collection!',
                      name, module.name(), module.version())
