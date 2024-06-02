import shutil

from typing import List, Optional, Tuple

from .split import stratified_train_test_split


def indices_to_file(indices: List[int], file_name: str) -> None:
    with open(f'{file_name}', 'w') as f:
        for index in indices:
            f.write(str((int(index))) + '\n')


def indices_from_file(file_name: str) -> List[int]:
    indices = []
    with open(file_name, 'r') as f:
        for line in f:
            index = int(line.strip())
            indices.append(index)
    return indices


def test_train_indices(
    indices: List[int],
    labels: List[str],
    output_dir: str,
    test_indices_file: Optional[str] = None,
    train_ratio: Optional[float] = None,
    random_seed: Optional[int] = None
) -> Tuple[List[int], List[int]]:
    test_indices = []
    train_indices = []

    if test_indices_file is not None:
        # copy test indices into output dir
        shutil.copyfile(test_indices_file, f'{output_dir}/test_indices.txt')

        # load test indices from file
        test_indices = indices_from_file(f'{output_dir}/test_indices.txt')

        # create trian_indices by removing loaded indices form indices list
        train_indices = [index for index in indices if index not in test_indices]

    elif train_ratio is not None:
        # split the list of indices into train and test
        train_indices, test_indices = stratified_train_test_split(
            indices,
            labels,
            train_size=train_ratio,
            random_state=random_seed
        )

        indices_to_file(test_indices, f'{output_dir}/test_indices.txt')

    return test_indices, train_indices
