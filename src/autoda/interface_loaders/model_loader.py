import logging

from ..api.model_interface import ModelInterface
from ..globals import MODEL_FACTORY_NAME
from ..factories import get_factory

_LOG = logging.getLogger(__name__)


class ModelLoader:
    def load(self, module: ModelInterface) -> None:
        models = module.models()
        for name in models:
            get_factory(MODEL_FACTORY_NAME).register(name, models[name])
            _LOG.info('Model \'%s\' from %s (%s) added to model_factory!',
                      name, module.name(), module.version())
