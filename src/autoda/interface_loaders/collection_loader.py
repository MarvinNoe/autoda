import logging

from ..api import CollectionInterface
from ..collections import register_collection


_LOG = logging.getLogger(__name__)


class CollectionLoader:
    def load(self, module: CollectionInterface) -> None:
        collections = module.collections()
        for name in collections:
            register_collection(name, collections[name])
            _LOG.info('Collection \'%s\' from %s (%s) added to collections!',
                      name, module.name(), module.version())
