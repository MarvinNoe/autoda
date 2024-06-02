import logging

from ..api.routine_interface import RoutineInterface
from ..globals import ROUTINE_FACTORY_NAME
from ..factories import get_factory


_LOG = logging.getLogger(__name__)


class RoutineLoader:
    def load(self, module: RoutineInterface) -> None:
        routines = module.routines()
        for name in routines:
            get_factory(ROUTINE_FACTORY_NAME).register(name, routines[name])
            _LOG.info('Routine \'%s\' from %s (%s) added to routine_factory!',
                      name, module.name(), module.version())
