from typing import Tuple, List, Optional, Any, Callable

import sklearn.model_selection as sk

from torch.utils import data

from .labels import normalized_label_ids, unique_label_ids


def train_test_split(
    indices: List[int],
    test_size: Optional[float] = None,
    train_size: Optional[float] = None,
    random_state: Optional[int] = None,
    shuffle: Optional[bool] = True
) -> Any:
    """
    Splits the specified list of indices into training and test sets. It uses
    sklearn.model_selection.train_test_split() for the split. In contrast to the train_test_split()
    of sklearn, this method also accepts lists that have less than 2 elements.
    If the list contains only one element, the element is part of the training set
    and the test set will be empty.

    - `indices: List[int]` List of indices that should be split.

    - `test_size: Optional[float] = None` If float, should be between 0.0 and 1.0 and represent the
        proportion of the dataset to include in the test split. If int, represents the absolute
        number of test samples. If None, the value is set to the complement of the train size.
        If train_size is also None, it will be set to 0.25.

    - `train_size: Optional[float] = None` If float, should be between 0.0 and 1.0 and represent
        the proportion of the dataset to include in the train split. If int, represents the
        absolute number of train samples. If None, the value is automatically set to the complement
        of the test size.

    - `random_state: Optional[int] = None` Controls the shuffling applied to the data before
        applying the split. Pass an int for reproducible output across multiple function calls.
        See [Glossary](https://scikit-learn.org/stable/glossary.html#term-random_state).

    - `shuffle: Optional[bool] = Whether or not to shuffle the data before splitting.
        If shuffle=False then stratify must be None.

    Returns two lists. The first list contains the indices of the training samples and the second
    contains the indices of the test samples.
    """
    if len(indices) == 1:
        return indices[:], []
    if len(indices) > 1:
        return sk.train_test_split(
            indices,
            test_size=test_size,
            train_size=train_size,
            random_state=random_state,
            shuffle=shuffle
        )
    return [], []


def stratified_train_test_split(
    indices: List[int],
    labels: List[str],
    test_size: Optional[float] = None,
    train_size: Optional[float] = None,
    random_state: Optional[int] = None
) -> Tuple[List[int], List[int]]:
    """
    Splits the specified indices stratified into test and training sets. The stratification is
    based on the specified labels. It uses sklearn.model_selection.train_test_split() for the split.
    In contrast to the train_test_split() of sklearn it splits the labels even if there are labels
    that only occurs once.

    - `indices: List[int]` List of indices that should be split.

    - `labels: List[str]` List of labels that should be used for stratification.

    - `test_size: Optional[float] = None` If float, should be between 0.0 and 1.0 and represent the
        proportion of the dataset to include in the test split. If int, represents the absolute
        number of test samples. If None, the value is set to the complement of the train size.
        If train_size is also None, it will be set to 0.25.

    - `train_size: Optional[float] = None` If float, should be between 0.0 and 1.0 and represent
        the proportion of the dataset to include in the train split. If int, represents the
        absolute number of train samples. If None, the value is automatically set to the complement
        of the test size.

    - `random_state: Optional[int] = None` Controls the shuffling applied to the data before
        applying the split. Pass an int for reproducible output across multiple function calls.
        See [Glossary](https://scikit-learn.org/stable/glossary.html#term-random_state).

    Returns two lists. The first list contains the indices of the training samples and the second
    contains the indices of the test samples.
    """
    indices_copy = indices.copy()
    labels_copy = labels.copy()

    unique_ids = unique_label_ids(labels_copy)
    unique_id_indices = []

    # remove unique label IDs and the corresponding indices from
    # the list of IDs and indices
    for unique_id in unique_ids:
        index = labels_copy.index(unique_id)
        del labels_copy[index]
        del indices_copy[index]
        unique_id_indices.append(index)

    # stratified split
    train_indices, test_indices = sk.train_test_split(
        indices_copy,
        test_size=test_size,
        train_size=train_size,
        random_state=random_state,
        shuffle=True,
        stratify=labels_copy
    )

    # split unique labels
    train_indices_u, test_indices_u = train_test_split(
        unique_id_indices,
        test_size=test_size,
        train_size=train_size,
        random_state=random_state,
        shuffle=True
    )

    return train_indices + train_indices_u, \
        test_indices + test_indices_u


def stratified_dataset_split(
    dataset: data.Dataset[Tuple[Any, Any]],
    test_size: Optional[float] = None,
    train_size: Optional[float] = None,
    random_state: Optional[int] = None,
    label_ids_from_target: Optional[Callable[[Any], List[int]]] = None
) -> Tuple[List[int], List[int]]:
    """
    Reads the label IDs of the specified dataset and splits them stratified into test and
    training sets. It uses sklearn.model_selection.train_test_split() for the split. In contrast to
    the train_test_split() of sklearn it splits the labels even if there are labels that only
    occurs once.

    - `dataset: torch.utils.data.Dataset` PyTorch dataset that sould be split into train and test
        sets.

    - `test_size: Optional[float] = None` If float, should be between 0.0 and 1.0 and represent the
        proportion of the dataset to include in the test split. If int, represents the absolute
        number of test samples. If None, the value is set to the complement of the train size.
        If train_size is also None, it will be set to 0.25.

    - `train_size: Optional[float] = None` If float, should be between 0.0 and 1.0 and represent
        the proportion of the dataset to include in the train split. If int, represents the
        absolute number of train samples. If None, the value is automatically set to the complement
        of the test size.

    - `random_state: Optional[int] = None` Controls the shuffling applied to the data before
        applying the split. Pass an int for reproducible output across multiple function calls.
        See [Glossary](https://scikit-learn.org/stable/glossary.html#term-random_state).

    - `label_from_target: Callable[[Any], int] = None` Callback to get the label IDs from a target.
        If it is None, it assumes that the target is compatible with Torchvisions
        transforms v2. This argument can be used to define a custom callback that gets a target
        from the specified dataset and returns the label IDs as a list of integers.

    Returns two lists. The first list contains the indices of the training samples and the second
    contains the indices of the test samples.
    """
    normalized_ids = normalized_label_ids(dataset, label_ids_from_target=label_ids_from_target)
    indices = list(range(len(normalized_ids)))

    return stratified_train_test_split(
        indices,
        normalized_ids,
        test_size=test_size,
        train_size=train_size,
        random_state=random_state
    )
