import torch
from torch.utils import data

from autoda.data import split


def test_train_test_split_empty_list() -> None:
    indices = []
    train, test = split.train_test_split(indices)

    assert len(train) == 0
    assert len(test) == 0


def test_train_test_split_one_element() -> None:
    indices = [0]
    train, test = split.train_test_split(indices)

    assert len(train) == 1
    assert train[0] == 0
    assert len(test) == 0


def test_train_test_split_default_ratio() -> None:
    indices = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]
    train, test = split.train_test_split(indices)

    assert len(train) == 9
    assert len(test) == 3


def test_train_test_split_test_size() -> None:
    indices = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]
    train, test = split.train_test_split(indices, test_size=0.3)

    assert len(train) == 8
    assert len(test) == 4


def test_train_test_split_train_size() -> None:
    indices = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]
    train, test = split.train_test_split(indices, train_size=0.7)

    assert len(train) == 8
    assert len(test) == 4


def is_sorted(lst):
    return all(lst[i] <= lst[i + 1] for i in range(len(lst) - 1))


def test_train_test_split_shuffle_true():
    indices = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]
    train, test = split.train_test_split(indices, train_size=0.5)

    assert not is_sorted(train)
    assert not is_sorted(test)


def test_train_test_split_shuffle_false():
    indices = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]
    train, test = split.train_test_split(indices, train_size=0.5, shuffle=False)

    assert is_sorted(train)
    assert is_sorted(test)


def test_stratified_train_test_split_default_ratio() -> None:
    indices = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]
    labels = ['1', '1', '1', '1', '1', '1', '2', '2', '2', '2', '1-1-2', '1-2-2']
    train, test = split.stratified_train_test_split(indices, labels)

    assert len(train) == 8
    assert len(test) == 4


def test_stratified_train_test_split_test_size() -> None:
    indices = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]
    labels = ['1', '1', '1', '1', '1', '1', '2', '2', '2', '2', '1-1-2', '1-2-2']
    train, test = split.stratified_train_test_split(indices, labels, test_size=0.5)

    assert len(train) == 6
    assert len(test) == 6


def test_stratified_train_test_split_train_size() -> None:
    indices = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]
    labels = ['1', '1', '1', '1', '1', '1', '2', '2', '2', '2', '1-1-2', '1-2-2']
    train, test = split.stratified_train_test_split(indices, labels, train_size=0.5)

    assert len(train) == 6
    assert len(test) == 6


class CustomDatasetList(data.Dataset):
    def __init__(self):
        self.data = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l']
        self.targets = [
            {'labels': [1]},
            {'labels': [1]},
            {'labels': [1]},
            {'labels': [1]},
            {'labels': [1]},
            {'labels': [1]},
            {'labels': [2]},
            {'labels': [2]},
            {'labels': [2]},
            {'labels': [2]},
            {'labels': [1, 1, 2]},
            {'labels': [1, 2, 2]}
        ]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx], self.targets[idx]


class CustomDatasetTensor(data.Dataset):
    def __init__(self):
        self.data = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l']
        self.targets = [
            {'labels': torch.tensor([1])},
            {'labels': torch.tensor([1])},
            {'labels': torch.tensor([1])},
            {'labels': torch.tensor([1])},
            {'labels': torch.tensor([1])},
            {'labels': torch.tensor([1])},
            {'labels': torch.tensor([2])},
            {'labels': torch.tensor([2])},
            {'labels': torch.tensor([2])},
            {'labels': torch.tensor([2])},
            {'labels': torch.tensor([1, 1, 2])},
            {'labels': torch.tensor([1, 2, 2])}
        ]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx], self.targets[idx]


def test_stratified_dataset_split_list() -> None:
    dataset = CustomDatasetList()

    def get_label_from_target(target):
        return target['labels']

    train, test = split.stratified_dataset_split(
        dataset,
        label_ids_from_target=get_label_from_target
    )

    assert len(train) == 9
    assert len(test) == 3


def test_stratified_dataset_split_tensor() -> None:
    dataset = CustomDatasetTensor()
    train, test = split.stratified_dataset_split(dataset)

    assert len(train) == 9
    assert len(test) == 3
