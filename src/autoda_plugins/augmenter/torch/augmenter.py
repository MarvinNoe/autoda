import copy

from typing import List, Callable, Any, Tuple

import pandas as pd
import torch

from torch.utils import data
from torchvision.datasets import vision
from torchvision.transforms import v2

from ...fastaa.policy_decode import decode_fastaa_policy_df
from ...smart_augment import smart_agument as sa
from ...trivial_augment.transform import TrivialAugment


def fastaa_augmented_dataset(
        dataset: vision.VisionDataset,
        indices: List[int],
        labels: List[str],
        policy_csv_file: str,
        transforms_creator: Callable[..., Any],
        **kwargs
) -> Tuple[data.Dataset, List[str]]:
    # copy dataset
    aug_dataset = copy.deepcopy(dataset)

    # get transforms from CSV file
    df = pd.read_csv(policy_csv_file)
    fastaa_policy = decode_fastaa_policy_df(df)
    fastaa_transforms_v2 = fastaa_policy.torchvision_random_choice_v2(transforms_creator())
    fastaa_transforms_v2 = v2.Compose([fastaa_transforms_v2, v2.SanitizeBoundingBoxes()])

    # set transforms
    if aug_dataset.transforms is not None:
        aug_dataset.transforms = v2.Compose([aug_dataset.transforms, fastaa_transforms_v2])
    else:
        aug_dataset.transforms = fastaa_transforms_v2

    # concatenate datasets
    # return data.ConcatDataset([data.Subset(dataset, indices), data.Subset(aug_dataset, indices)])
    return data.Subset(aug_dataset, indices), labels


def extended_fastaa_augmented_dataset(
        dataset: vision.VisionDataset,
        indices: List[int],
        labels: List[str],
        policy_csv_file: str,
        transforms_creator: Callable[..., Any],
        **kwargs
) -> Tuple[data.Dataset, List[str]]:
    # copy dataset
    aug_dataset = copy.deepcopy(dataset)

    # get transforms from CSV file
    df = pd.read_csv(policy_csv_file)
    fastaa_policy = decode_fastaa_policy_df(df)
    fastaa_transforms_v2 = fastaa_policy.torchvision_random_choice_v2(transforms_creator())
    fastaa_transforms_v2 = v2.Compose([fastaa_transforms_v2, v2.SanitizeBoundingBoxes()])

    # set transforms
    if aug_dataset.transforms is not None:
        aug_dataset.transforms = v2.Compose([aug_dataset.transforms, fastaa_transforms_v2])
    else:
        aug_dataset.transforms = fastaa_transforms_v2

    extended_labels = labels + labels

    # concatenate datasets
    return data.ConcatDataset(
        [
            data.Subset(dataset, indices),
            data.Subset(aug_dataset, indices)
        ]
    ), extended_labels


def extended_fastaa_augmented_dataset_full(
        dataset: vision.VisionDataset,
        indices: List[int],
        labels: List[str],
        policy_csv_file: str,
        transforms_creator: Callable[..., Any],
        **kwargs
) -> Tuple[data.Dataset, List[str]]:
    # copy dataset
    aug_dataset = copy.deepcopy(dataset)

    # get transforms from CSV file
    df = pd.read_csv(policy_csv_file)
    fastaa_policy = decode_fastaa_policy_df(df)
    fastaa_transforms_v2 = fastaa_policy.torchvision_random_choice_v2(transforms_creator())
    fastaa_transforms_v2 = v2.Compose([fastaa_transforms_v2, v2.SanitizeBoundingBoxes()])

    # set transforms
    if aug_dataset.transforms is not None:
        aug_dataset.transforms = v2.Compose([aug_dataset.transforms, fastaa_transforms_v2])
    else:
        aug_dataset.transforms = fastaa_transforms_v2

    extended_labels = labels + labels

    # concatenate datasets
    return data.ConcatDataset(
        [
            data.Subset(aug_dataset, indices),
            data.Subset(aug_dataset, indices)
        ]
    ), extended_labels


def jaccard_overlap_augmented_dataset(
        dataset: vision.VisionDataset,
        indices: List[int],
        labels: List[str],
        min_scale=0.1,
        max_scale=1.0,
        min_aspect_ratio=0.5,
        max_aspect_ratio=2.0,
        sampler_options=[0.0, 0.1, 0.3, 0.5, 0.7, 0.9, 1.0],
        trials=40,
        ** kwargs
) -> Tuple[data.Dataset, List[str]]:
    # copy dataset
    aug_dataset = copy.deepcopy(dataset)

    #  minimum Jaccard overlap transformation
    jaccard_transforms = v2.Compose(
        [
            v2.ToImage(),
            v2.RandomIoUCrop(
                min_scale=min_scale,
                max_scale=max_scale,
                min_aspect_ratio=min_aspect_ratio,
                max_aspect_ratio=max_aspect_ratio,
                sampler_options=sampler_options,
                trials=trials
            ),
            v2.SanitizeBoundingBoxes(),
            v2.ToDtype(torch.float32, scale=True),
        ]
    )

    # set transforms
    if aug_dataset.transforms is not None:
        aug_dataset.transforms = v2.Compose([aug_dataset.transforms, jaccard_transforms])
    else:
        aug_dataset.transforms = jaccard_transforms

    # return subset of augmented dataset
    return data.Subset(aug_dataset, indices), labels


def extended_jaccard_overlap_augmented_dataset(
        dataset: vision.VisionDataset,
        indices: List[int],
        labels: List[str],
        min_scale=0.1,
        max_scale=1.0,
        min_aspect_ratio=0.5,
        max_aspect_ratio=2.0,
        sampler_options=[0.0, 0.1, 0.3, 0.5, 0.7, 0.9, 1.0],
        trials=40,
        ** kwargs
) -> Tuple[data.Dataset, List[str]]:
    # copy dataset
    aug_dataset = copy.deepcopy(dataset)

    #  minimum Jaccard overlap transformation
    jaccard_transforms = v2.Compose(
        [
            v2.ToImage(),
            v2.RandomIoUCrop(
                min_scale=min_scale,
                max_scale=max_scale,
                min_aspect_ratio=min_aspect_ratio,
                max_aspect_ratio=max_aspect_ratio,
                sampler_options=sampler_options,
                trials=trials
            ),
            v2.SanitizeBoundingBoxes(),
            v2.ToDtype(torch.float32, scale=True),
        ]
    )

    # set transforms
    if aug_dataset.transforms is not None:
        aug_dataset.transforms = v2.Compose([aug_dataset.transforms, jaccard_transforms])
    else:
        aug_dataset.transforms = jaccard_transforms

    extended_labels = labels + labels

    # concatenate datasets
    return data.ConcatDataset(
        [
            data.Subset(dataset, indices),
            data.Subset(aug_dataset, indices)
        ]
    ), extended_labels


def trivial_augmented_dataset(
        dataset: vision.VisionDataset,
        indices: List[int],
        labels: List[str],
        transforms_creator: Callable[..., Any],
        **kwargs
) -> Tuple[data.Dataset, List[str]]:
    # copy dataset
    aug_dataset = copy.deepcopy(dataset)

    trivial_augment = TrivialAugment(transforms_creator())
    trivial_augment = v2.Compose([trivial_augment, v2.SanitizeBoundingBoxes()])

    # set transforms
    if aug_dataset.transforms is not None:
        aug_dataset.transforms = v2.Compose([aug_dataset.transforms, trivial_augment])
    else:
        aug_dataset.transforms = trivial_augment

    # return subset of augmented dataset
    return data.Subset(aug_dataset, indices), labels


def extended_trivial_augmented_dataset(
        dataset: vision.VisionDataset,
        indices: List[int],
        labels: List[str],
        transforms_creator: Callable[..., Any],
        **kwargs
) -> Tuple[data.Dataset, List[str]]:
    # copy dataset
    aug_dataset = copy.deepcopy(dataset)

    trivial_augment = TrivialAugment(transforms_creator())
    trivial_augment = v2.Compose([trivial_augment, v2.SanitizeBoundingBoxes()])

    # set transforms
    if aug_dataset.transforms is not None:
        aug_dataset.transforms = v2.Compose([aug_dataset.transforms, trivial_augment])
    else:
        aug_dataset.transforms = trivial_augment

    extended_labels = labels + labels

    # concatenate datasets
    return data.ConcatDataset(
        [
            data.Subset(dataset, indices),
            data.Subset(aug_dataset, indices)
        ]
    ), extended_labels


def extended_three_trivial_augmented_dataset(
        dataset: vision.VisionDataset,
        indices: List[int],
        labels: List[str],
        transforms_creator: Callable[..., Any],
        **kwargs
) -> Tuple[data.Dataset, List[str]]:
    # copy dataset
    aug_dataset = copy.deepcopy(dataset)

    trivial_augment = TrivialAugment(transforms_creator())
    trivial_augment = v2.Compose([trivial_augment, v2.SanitizeBoundingBoxes()])

    # set transforms
    if aug_dataset.transforms is not None:
        aug_dataset.transforms = v2.Compose([aug_dataset.transforms, trivial_augment])
    else:
        aug_dataset.transforms = trivial_augment

    extended_labels = labels + labels + labels

    # concatenate datasets
    return data.ConcatDataset(
        [
            data.Subset(dataset, indices),
            data.Subset(aug_dataset, indices),
            data.Subset(aug_dataset, indices)
        ]
    ), extended_labels


def extended_full_trivial_augmented_dataset(
        dataset: vision.VisionDataset,
        indices: List[int],
        labels: List[str],
        transforms_creator: Callable[..., Any],
        **kwargs
) -> Tuple[data.Dataset, List[str]]:
    # copy dataset
    aug_dataset = copy.deepcopy(dataset)

    trivial_augment = TrivialAugment(transforms_creator())
    trivial_augment = v2.Compose([trivial_augment, v2.SanitizeBoundingBoxes()])

    # set transforms
    if aug_dataset.transforms is not None:
        aug_dataset.transforms = v2.Compose([aug_dataset.transforms, trivial_augment])
    else:
        aug_dataset.transforms = trivial_augment

    extended_labels = labels + labels

    # concatenate datasets
    return data.ConcatDataset(
        [
            data.Subset(aug_dataset, indices),
            data.Subset(aug_dataset, indices)
        ]
    ), extended_labels


def smart_augmented_dataset(
    dataset: vision.VisionDataset,
    indices: List[int],
    labels: List[str],
    level_col: float,
    level_geo: float,
    num_col_trans: int,
    num_geo_trans: int,
    prob: float,
    color_transforms: Callable[..., v2.Transform],
    geometry_transforms: Callable[..., v2.Transform],
    ** kwargs
) -> Tuple[data.Dataset, List[str]]:
    return sa.smart_augmented_dataset(
        dataset=dataset,
        indices=indices,
        level_col=level_col,
        level_geo=level_geo,
        num_col_trans=num_col_trans,
        num_geo_trans=num_geo_trans,
        prob=prob,
        color_transforms=color_transforms,
        geometry_transforms=geometry_transforms
    ), labels


def extended_smart_augmented_dataset(
    dataset: vision.VisionDataset,
    indices: List[int],
    labels: List[str],
    level_col: float,
    level_geo: float,
    num_col_trans: int,
    num_geo_trans: int,
    prob: float,
    color_transforms: Callable[..., v2.Transform],
    geometry_transforms: Callable[..., v2.Transform],
    ** kwargs
) -> Tuple[data.Dataset, List[str]]:
    aug_dataset = sa.smart_augmented_dataset(
        dataset=dataset,
        indices=indices,
        level_col=level_col,
        level_geo=level_geo,
        num_col_trans=num_col_trans,
        num_geo_trans=num_geo_trans,
        prob=prob,
        color_transforms=color_transforms,
        geometry_transforms=geometry_transforms
    )

    extended_labels = labels + labels

    # concatenate datasets
    return data.ConcatDataset(
        [
            data.Subset(dataset, indices),
            aug_dataset
        ]
    ), extended_labels


def extended_doubler(
    dataset: vision.VisionDataset,
    indices: List[int],
    labels: List[str],
    ** kwargs
) -> Tuple[data.Dataset, List[str]]:

    sec_dataset = copy.deepcopy(dataset)

    extended_labels = labels + labels

    # concatenate datasets
    return data.ConcatDataset(
        [
            data.Subset(dataset, indices),
            data.Subset(sec_dataset, indices)
        ]
    ), extended_labels
