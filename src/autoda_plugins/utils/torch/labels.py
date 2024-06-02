from typing import Tuple, List, Callable, Any, Optional, Dict

import numpy as np
from torch.utils import data

from tqdm import tqdm


def _label_ids_from_dataloader_target(target: Dict[str, Any]) -> Any:
    labels = target['labels'].squeeze().tolist()
    return labels if isinstance(labels, list) else [labels]


def label_ids(
    dataset: data.Dataset[Tuple[Any, Any]],
    label_ids_from_target: Optional[Callable[[Any], List[int]]] = None
) -> List[str]:
    """
    Returns the label IDs for all entries in the dataset as a list of strings.
    This function converts the label IDs into strings. For entries with multiple labels, the label
    IDs are concatenated with a "-" as a separator.

    - `dataset: torch.utils.data.Dataset` PyTorch dataset from which the label IDs should
        be extracted.

    - `label_from_target: Callable[[Any], int] = None` Callback to get the label IDs from a target.
        If it is None, it assumes that the target is compatible with Torchvisions
        transforms v2. This argument can be used to define a custom callback that gets a target
        from the specified dataset and returns the label IDs as a list of integers.

    Returns the list of label IDs.
    """
    if label_ids_from_target is None:
        label_ids_from_target = _label_ids_from_dataloader_target

    labels = []

    tmp_loader = data.DataLoader(dataset)
    prog_bar = tqdm(tmp_loader, total=len(tmp_loader))
    prog_bar.set_description(desc='Label normalization')

    # get all labels from the targets of the dataset and concatenate multiple labels
    for _, target in prog_bar:
        labels.append('-'.join([str(x) for x in label_ids_from_target(target)]))

    return labels


def normalized_label_ids(
    dataset: data.Dataset[Tuple[Any, Any]],
    label_ids_from_target: Optional[Callable[[Any], List[int]]] = None
) -> List[str]:
    """
    Returns the label IDs for all entries in the dataset as a list of normalized strings.
    This means that if the entries are multi-labeled, duplicates are removed, the labels are sorted,
    converted into strings and concatenated separated by "-".

    - `dataset: torch.utils.data.Dataset` PyTorch dataset from which the label IDs should
        be extracted.

    - `label_from_target: Callable[[Any], int] = None` Callback to get the label IDs from a target.
        If it is None, it assumes that the target is compatible with Torchvisions
        transforms v2. This argument can be used to define a custom callback that gets a target
        from the specified dataset and returns the label IDs as a list of integers.

    Returns the list of normalized and string converted labels.
    """

    if label_ids_from_target is None:
        label_ids_from_target = _label_ids_from_dataloader_target

    def normalized(target: Dict[str, Any]) -> Any:
        return sorted(set(label_ids_from_target(target)))

    return label_ids(dataset, label_ids_from_target=normalized)


def unique_label_ids(label_ids: List[str]) -> List[str]:
    """
    Returns the label IDs that only occur once in the specified list of label IDs.

    - `label_ids: List[str]` List of label IDs.

    Returns the list of unique label IDs.
    """
    id, count = np.unique(label_ids, return_counts=True)
    return [id[i] for i in np.where(count == 1)[0]]
