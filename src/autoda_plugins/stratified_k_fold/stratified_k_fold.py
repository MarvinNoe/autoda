
import copy
import logging
import os
import time

from typing import List, Tuple, Any, Dict, Optional

import numpy as np
import sklearn.model_selection as sk
import torch
import torch.optim as optim

from torch._prims_common import DeviceLikeType
from torch.nn.parameter import Parameter
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LRScheduler
from torch.utils import data
from torchmetrics.detection import mean_ap
from torchvision.transforms import v2

import autoda.globals as autoda_globals

from autoda.abstract import Routine
from autoda.config.config_object import ConfigObject
from autoda.config.yaml_loader import load_config

from .trainer import Trainer
from .detection_validator import DetectionValidator
from ..plugins.globals import AUGMENTER_FAC_NAME as AUGMENTER
from ..utils.random_setup import set_random_seed
from ..utils.torch.save import SaveBestModel, save_model, save_loss_plot, save_m_ap
from ..utils.torch.indices import indices_to_file, test_train_indices
from ..utils.torch.labels import normalized_label_ids

_LOG = logging.getLogger(__name__)

MODEL = autoda_globals.MODEL_FACTORY_NAME
DATASET = autoda_globals.DATASET_FACTORY_NAME


def values_to_file(value_list: List[float], save_name: str) -> None:
    with open(f'{save_name}.txt', 'w') as f:
        for value in value_list:
            f.write(str(value) + '\n')


def collate_fn(batch: Any) -> Any:
    """
    To handle the data loading as different images may have different number
    of objects and to handle varying size tensors as well.
    """
    return tuple(zip(*batch))


def calculate_sample_weights(labels: List[str]) -> List[Any]:
    # Compute class frequencies
    class_frequencies: Dict[str, int] = {}
    for sample_labels in labels:
        for label in sample_labels.split('-'):
            if label in class_frequencies:
                class_frequencies[label] += 1
            else:
                class_frequencies[label] = 1

    # Compute class weights
    total_samples = len(labels)
    class_weights = {label: total_samples / frequency for label,
                     frequency in class_frequencies.items()}

    # Compute sample weights
    sample_weights = []
    for sample_labels in labels:
        weight = np.mean([class_weights[label] for label in sample_labels.split('-')])
        sample_weights.append(weight)

    # Normalize sample weights to sum to 1
    sample_weights /= np.sum(sample_weights)
    return sample_weights


def sgd_optimizer(params: List[Parameter]) -> Tuple[Optimizer, LRScheduler]:
    optimizer = optim.SGD(
        params,
        lr=0.0001,
        weight_decay=0.00005,
        momentum=0.9
    )
    scheduler = optim.lr_scheduler.StepLR(
        optimizer=optimizer,
        step_size=15,
        gamma=0.1
    )

    return optimizer, scheduler


def create_augmented_dataset(dataset_config: ConfigObject, policy: v2.Transform) -> data.Dataset:
    dataset_config_copy = copy.deepcopy(dataset_config)
    transform_policy = policy

    try:
        transforms_config = dataset_config_copy.get_arg('transforms')
    except KeyError:
        transforms_config = None

    if transforms_config is not None:
        default_transforms = transforms_config.create_instance()
        transform_policy = v2.Compose([default_transforms, transform_policy])

    dataset_config_copy.set_arg('transforms', transform_policy)

    return dataset_config_copy.create_instance()


def train_and_eval(
    *,
    fold: int,
    train_indices: List[int],
    val_indices: List[int],
    train_labels: List[str],
    config: Dict[str, ConfigObject],
    epochs: int,
    batch_size: int,
    num_workers: int,
    device: DeviceLikeType,
    output_dir: str,
    random_seed: Optional[int] = None,
    model_name_prefix: str = 'best_model'
) -> None:

    # initialize the trainer and the Validator
    trainer = Trainer()
    validator = DetectionValidator()

    save_best_model_of_fold = SaveBestModel()

    # create model
    model = config[MODEL].create_instance()
    model.to(device)

    # for efficient execution on the gcp cluster
    # model.cuda()
    # model = torch.nn.parallel.DataParallel(model, device_ids=list(range(4)), dim=0)

    # create optimizer and scheduler
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer, scheduler = sgd_optimizer(params)

    train_dataset = config[DATASET].create_instance()
    labels = train_labels

    # create data set
    if AUGMENTER in config:
        _LOG.info('%s is used to create augmented training dataset.', config[AUGMENTER].type_key)
        train_dataset, labels = config[AUGMENTER].create_instance(
            dataset=train_dataset,
            indices=train_indices,
            labels=train_labels
        )
    else:
        _LOG.info('No augmentation is used!')
        train_dataset = data.Subset(train_dataset, train_indices)

    # TODO: if the augmentation is based on randomness, normalization makes no sense
    # labels = normalized_label_ids(train_dataset)

    # sampler
    sample_weights = calculate_sample_weights(labels)
    sampler = data.WeightedRandomSampler(sample_weights, len(sample_weights))

    # use train_indices and val_indices to create DataLoader instances
    train_loader = data.DataLoader(
        train_dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        collate_fn=collate_fn,
        drop_last=True,
        pin_memory=True,
        sampler=sampler
    )

    val_loader = data.DataLoader(
        data.Subset(config[DATASET].create_instance(), val_indices),
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=collate_fn,
        drop_last=True,
        pin_memory=True
    )

    # To store training loss and mAP values.
    train_loss_list = []
    map_50_list = []
    map_list = []
    best_map = 0.
    patience_counter = 0
    patience = 5

    # Training loop.
    for epoch in range(epochs):
        _LOG.info('[Fold: %s, Epoch: %s] Start epoch %s of %s', fold, epoch+1, epoch+1, epochs)

        # Start timer and carry out training and validation.
        start = time.time()
        train_loss, average_loss = trainer.run(
            model=model,
            dataloader=train_loader,
            optimizer=optimizer,
            device=device,
            progress=False
        )
        target, pred = validator.run(
            model=model,
            dataloader=val_loader,
            device=device,
            progress=False
        )

        metric = mean_ap.MeanAveragePrecision(class_metrics=False)
        metric.update(pred, target)
        metric_summary = metric.compute()

        _LOG.info('[Fold: %s, Epoch: %s] last validation loss: %.3f', fold, epoch+1, train_loss)
        _LOG.info('[Fold: %s, Epoch: %s] average train loss: %.3f', fold, epoch+1, average_loss)
        _LOG.info('[Fold: %s, Epoch: %s] mAP %s', fold, epoch+1, metric_summary['map'].item())

        train_loss_list.append(train_loss)
        map_50_list.append(metric_summary['map_50'])
        map_list.append(metric_summary['map'])
        end = time.time()

        _LOG.info('[Fold: %s, Epoch: %s] Took %.3f minutes', fold, epoch+1, ((end - start) / 60))

        model.cpu()

        # save the best model till now.
        save_best_model_of_fold(
            model,
            float(metric_summary['map']),
            epoch,
            output_dir,
            f'{model_name_prefix}_fold_{fold}'
        )

        # Save the current epoch model.
        save_model(epoch, model, optimizer)

        # Save loss plot.
        save_loss_plot(output_dir, train_loss_list, save_name=f'train_loss_fold_{fold}')

        # Save mAP plot.
        save_m_ap(output_dir, map_50_list, map_list, save_name=f'map_{fold}')

        _LOG.info(
            '[Fold: %s, Epoch: %s] last lerning rate: %s', fold, epoch+1, scheduler.get_last_lr()
        )

        scheduler.step()

        # Early Stopping
        if metric_summary['map'] > best_map:
            best_map = metric_summary['map']
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                _LOG.info('[Fold: %s, Epoch: %s] Early stopping!', fold, epoch+1)
                break

    values_to_file(map_list, f'{output_dir}/map_list_fold_{fold}')
    values_to_file(map_50_list, f'{output_dir}/map_50_list_fold_{fold}')
    values_to_file(train_loss_list, f'{output_dir}/train_loss_list_fold_{fold}')


def stratified_k_fold_cross_validation(
    *,
    folds: int,
    train_indices: List[int],
    train_labels: List[str],
    config: Dict[str, ConfigObject],
    epochs: int,
    batch_size: int,
    num_workers: int,
    device: DeviceLikeType,
    output_dir: str,
    random_seed: Optional[int] = None,
    model_name_prefix: str = 'best_model',
    val_indices_file_prefix: str = 'val_indices'
):
    # create an instance of StratifiedKFold
    skf = sk.StratifiedKFold(n_splits=folds, shuffle=True, random_state=random_seed)

    # iterate over each fold
    for fold, (fold_train_index, fold_val_index) in enumerate(
            skf.split(train_indices, train_labels)):

        # save validation indices of current fold
        indices_to_file(
            fold_val_index.tolist(),
            f'{output_dir}/{val_indices_file_prefix}_fold_{fold}.txt'
        )

        fold_train_labels = [train_labels[i] for i in fold_train_index]

        train_and_eval(
            fold=fold,
            train_indices=fold_train_index.tolist(),
            val_indices=fold_val_index.tolist(),
            train_labels=fold_train_labels,
            config=config,
            epochs=epochs,
            batch_size=batch_size,
            num_workers=num_workers,
            device=device,
            output_dir=output_dir,
            random_seed=random_seed,
            model_name_prefix=model_name_prefix
        )


class StratifiedKFoldRoutine(Routine):
    """
    Stratified K-Fold cross-validation routine.
    """

    def __init__(
        self,
        epochs: int,
        batch_size: int,
        num_workers: int,
        folds: int,
        device: str,
        output_dir: str,
        train_ratio: Optional[int] = None,
        test_indices_file: Optional[str] = None,
        random_seed: Optional[int] = None
    ):
        if (train_ratio is None and test_indices_file is None) or \
                (train_ratio is not None and test_indices_file is not None):
            raise ValueError("Exactly one of train_ratio or test_indices_file must be provided.")

        self.epochs = epochs
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.folds = folds
        self.device = torch.device(device)
        self.output_dir = output_dir
        self.train_ratio = train_ratio
        self.test_indices_file = test_indices_file
        self.random_seed = random_seed

    def exec(self, config_file: str) -> None:
        _LOG.info('Start k-fold Cross-Validation...')

        # set random seed
        set_random_seed(self.random_seed)

        # measure duration
        total_start = time.time()

        # load configuration file
        config = load_config(config_file)

        # create output directory
        os.makedirs(self.output_dir, exist_ok=True)

        # create dataset
        dataset = config[DATASET].create_instance()

        # get labels and indices of the data
        labels = normalized_label_ids(dataset)
        indices = list(range(len(dataset)))

        # split the list of indices into train and test
        test_indices = self.test_indices(indices, labels)

        # create lists for train indices and labels
        train_indices = [idx for idx in indices if idx not in test_indices]
        train_labels = [labels[idx] for idx in train_indices]

        stratified_k_fold_cross_validation(
            folds=self.folds,
            train_indices=train_indices,
            train_labels=train_labels,
            config=config,
            epochs=self.epochs,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            device=self.device,
            output_dir=self.output_dir,
            random_seed=self.random_seed
        )

        total_end = time.time()
        _LOG.info('Took %.3f minutes to execute k-fold Cross-Validation.',
                  ((total_end - total_start) / 60))

    def test_indices(self, indices: List[int], labels: List[str]) -> List[int]:
        test, _ = test_train_indices(
            indices,
            labels,
            self.output_dir,
            self.test_indices_file,
            self.train_ratio,
            self.random_seed
        )
        return test
