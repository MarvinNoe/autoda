from typing import Dict

from autoda.factories import AutoDaFactory

from .globals import AUGMENTER_FAC_NAME
from ..augmenter.torch.augmenter import (
    fastaa_augmented_dataset,
    extended_fastaa_augmented_dataset,
    extended_fastaa_augmented_dataset_full,
    jaccard_overlap_augmented_dataset,
    extended_jaccard_overlap_augmented_dataset,
    trivial_augmented_dataset,
    extended_trivial_augmented_dataset,
    extended_full_trivial_augmented_dataset,
    extended_three_trivial_augmented_dataset,
    smart_augmented_dataset,
    extended_smart_augmented_dataset,
    extended_doubler
)

_AUGMENTER_FAC = AutoDaFactory(AUGMENTER_FAC_NAME)

_AUGMENTER_FAC.register('torch.augmenter.fastaa', fastaa_augmented_dataset)
_AUGMENTER_FAC.register('torch.augmenter.extended.fastaa', extended_fastaa_augmented_dataset)
_AUGMENTER_FAC.register('torch.augmenter.extended.full.fastaa',
                        extended_fastaa_augmented_dataset_full)
_AUGMENTER_FAC.register('torch.augmenter.gc10', jaccard_overlap_augmented_dataset)
_AUGMENTER_FAC.register('torch.augmenter.extended.gc10', extended_jaccard_overlap_augmented_dataset)
_AUGMENTER_FAC.register('torch.augmenter.trivial_augment', trivial_augmented_dataset)
_AUGMENTER_FAC.register('torch.augmenter.extended.trivial_augment',
                        extended_trivial_augmented_dataset)
_AUGMENTER_FAC.register('torch.augmenter.extended.full.trivial_augment',
                        extended_full_trivial_augmented_dataset)
_AUGMENTER_FAC.register('torch.augmenter.extended.three.trivial_augment',
                        extended_three_trivial_augmented_dataset)
_AUGMENTER_FAC.register('torch.augmenter.smart_augment', smart_augmented_dataset)
_AUGMENTER_FAC.register('torch.augmenter.extended.smart_augment', extended_smart_augmented_dataset)

_AUGMENTER_FAC.register('torch.augmenter.extended.double', extended_doubler)


def name() -> str:
    """
    Returns the name of the factory plugin.
    """
    return "Factory Plugin"


def version() -> str:
    """
    Returns the version of the factory plugin.
    """
    return "0.1.0.dev0"


def factories() -> Dict[str, AutoDaFactory]:
    """
    Returns the factories provided by this plugin.
    """
    return {
        AUGMENTER_FAC_NAME: _AUGMENTER_FAC
    }
