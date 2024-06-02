from typing import Dict, Type

from autoda.abstract import Routine

from ..fastaa import FastAaRoutine
from ..smart_augment import SmartAugmentRoutine
from ..stratified_k_fold import StratifiedKFoldRoutine, ValidatedKFoldRoutine


def name() -> str:
    """
    Returns the name of the loadable component.
    """
    return "stratified K fold training routine"


def version() -> str:
    """
    Returns the version of the loadable component.
    """
    return "0.1.0.dev0"


def routines() -> Dict[str, Type[Routine]]:
    """
    Returns a dictionary mapping model names to corresponding modedl constructor.
    The returnd dictionary is added to the model factory.
    """
    return {
        "torch.routine.FastAA": FastAaRoutine,
        "torch.routine.SmartAugmentRoutine": SmartAugmentRoutine,
        "torch.train.detection.StratifiedKFold": StratifiedKFoldRoutine,
        "torch.validate.detection.ValidateKFold": ValidatedKFoldRoutine
    }
