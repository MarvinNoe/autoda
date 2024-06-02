from typing import Dict, Type

from .api import AutoDaInterface
from .interface_loaders import AutoDaLoader

from .api import CollectionInterface
from .interface_loaders import CollectionLoader

from .api import DatasetInterface
from .interface_loaders import DatasetLoader

from .api import FactoryInterface
from .interface_loaders import FactoryLoader

from .api import ModelInterface
from .interface_loaders import ModelLoader

from .api import RoutineInterface
from .interface_loaders import RoutineLoader

interface_loader_map: Dict[AutoDaInterface, Type[AutoDaLoader]] = {
    CollectionInterface: CollectionLoader,
    DatasetInterface: DatasetLoader,
    FactoryInterface: FactoryLoader,
    ModelInterface: ModelLoader,
    RoutineInterface: RoutineLoader
}
