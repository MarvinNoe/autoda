import copy
import logging
import os
import time

from typing import Optional, Callable, List, Tuple, Dict, Any

import ray
import torch
import torch.optim as optim

from ray import tune
from ray.tune import experiment
from ray.tune.search.hyperopt import HyperOptSearch
from ray.tune.search import ConcurrencyLimiter

from torch.nn.parameter import Parameter
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LRScheduler
from torch.utils import data
from torchvision.datasets import vision
from torchvision.transforms import v2
from torchmetrics.detection import mean_ap

import autoda.globals as autoda_globals

from autoda.abstract import Routine
from autoda.config.yaml_loader import load_config, load_plugins

from .globals import (
    PROB,
    NUM_COLOR_TRANSFORMS,
    NUM_GEOMETRY_TRANSFORMS,
    LEVEL_COLOR,
    LEVEL_GEOMETRY
)
from .space import smart_augment_default_search_space
from .transform import SmartAugmentTransform
from ..plugins.globals import SA_COLOR_TRANSF_COLL_NAME, SA_GEOMETRY_TRANSF_COLL_NAME
from ..utils.random_setup import set_random_seed
from ..utils.torch import split
from ..utils.torch.indices import test_train_indices
from ..utils.torch.labels import normalized_label_ids

from ..stratified_k_fold.stratified_k_fold import calculate_sample_weights, collate_fn
from ..stratified_k_fold.detection_validator import DetectionValidator
from ..stratified_k_fold.trainer import Trainer

_LOG: logging.Logger = logging.getLogger(__name__)

MODEL = autoda_globals.MODEL_FACTORY_NAME
DATASET = autoda_globals.DATASET_FACTORY_NAME
ROUTINE = autoda_globals.ROUTINE_FACTORY_NAME
COLOR_TRANSFORMS = SA_COLOR_TRANSF_COLL_NAME
GEO_TRANSFORMS = SA_GEOMETRY_TRANSF_COLL_NAME


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


def smart_augmented_dataset(
    dataset: vision.VisionDataset,
    indices: List[int],
    level_col: float,
    level_geo: float,
    num_col_trans: int,
    num_geo_trans: int,
    prob: float,
    color_transforms: Callable[..., v2.Transform],
    geometry_transforms: Callable[..., v2.Transform],
) -> data.Dataset:
    # copy dataset
    aug_dataset = copy.deepcopy(dataset)

    smart_augment = SmartAugmentTransform(
        col_transform_creators=color_transforms(),
        geo_transforms_creators=geometry_transforms(),
        level_col=level_col,
        level_geo=level_geo,
        num_col_trans=num_col_trans,
        num_geo_trans=num_geo_trans,
        prob=prob
    )

    smart_augment = v2.Compose([smart_augment, v2.SanitizeBoundingBoxes()])

    # set transforms
    if aug_dataset.transforms is not None:
        aug_dataset.transforms = v2.Compose([aug_dataset.transforms, smart_augment])
    else:
        aug_dataset.transforms = smart_augment

    # return subset of augmented dataset
    return data.Subset(aug_dataset, indices)


def trainable(
    config: Dict[str, Any],
    smart_augment_config_file: str,
    train_indices=None,
    train_labels=None,
    val_indices=None,
    epochs=None,
    dataset_path=None,
    batch_size=None,
    num_workers=None,
    device=None,
    random_seed=None
):
    # load plugins and configurations from file
    try:
        load_plugins(smart_augment_config_file)
    except KeyError:
        _LOG.info("Plugins already laoded.")

    autoda_config = load_config(smart_augment_config_file)

    # initialize the trainer and the Validator
    trainer = Trainer()
    validator = DetectionValidator()

    # create model
    model = autoda_config[MODEL].create_instance()
    model.to(device)

    # create optimizer and scheduler
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer, scheduler = sgd_optimizer(params)

    # create dataset
    autoda_config[DATASET].set_arg('root', dataset_path)
    dataset = autoda_config[DATASET].create_instance()

    color_transform_creators = autoda_config[ROUTINE].get_arg(COLOR_TRANSFORMS).create_instance()
    geo_transform_creators = autoda_config[ROUTINE].get_arg(GEO_TRANSFORMS).create_instance()

    # create data set
    train_dataset = smart_augmented_dataset(
        dataset=dataset,
        indices=train_indices,
        level_col=config[LEVEL_COLOR],
        level_geo=config[LEVEL_GEOMETRY],
        num_col_trans=config[NUM_COLOR_TRANSFORMS],
        num_geo_trans=config[NUM_GEOMETRY_TRANSFORMS],
        prob=config[PROB],
        color_transforms=color_transform_creators,
        geometry_transforms=geo_transform_creators
    )

    # TODO: if the augmentation is based on randomness, normalization makes no sense
    # labels = normalized_label_ids(train_dataset)
    labels = train_labels

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
        data.Subset(dataset, val_indices),
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=collate_fn,
        drop_last=True,
        pin_memory=True
    )

    # To store training loss and mAP values.
    best_map = torch.tensor(0.)
    # epoch_num = -1
    patience_counter = 0
    patience = 5

    total_start = time.time()

    # Training loop.
    for epoch in range(epochs):
        start = time.time()
        # Start timer and carry out training and validation.
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

        model.cpu()
        scheduler.step()

        end = time.time()
        _LOG.info('[Epoch: %s] Took %.3f minutes', epoch+1, ((end - start) / 60))
        print(f'[ID: {total_start}, Epoch: {epoch + 1}] Took {((end - start) / 60)} minutes')

        # Early Stopping
        if metric_summary['map'] > best_map:
            best_map = metric_summary['map']
            # epoch_num = epoch
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                _LOG.info('[Epoch: %s] Early stopping!', epoch+1)
                break

    return {'mAP': best_map.item()}


class SmartAugmentRoutine(Routine):
    def __init__(
        self,
        epochs: int,
        batch_size: int,
        num_samples: int,
        num_workers: int,
        max_concurrent: int,
        geometry_transforms: Callable[..., v2.Transform],
        color_transforms: Callable[..., v2.Transform],
        device: str,
        output_dir: str,
        save_name_prefix: str,
        val_ratio: float,  # Percentage of training data to be used for validation
        train_ratio: Optional[float] = None,
        test_indices_file: Optional[str] = None,
        random_seed: Optional[int] = None
    ):
        if (train_ratio is None and test_indices_file is None) or \
                (train_ratio is not None and test_indices_file is not None):
            raise ValueError("Exactly one of train_ratio or test_indices_file must be provided.")

        # input
        self.epochs = epochs
        self.batch_size = batch_size
        self.num_samples = num_samples
        self.num_workers = num_workers
        self.max_concurrent = max_concurrent
        self.geometry_transform_creators = geometry_transforms
        self.color_transform_creators = color_transforms
        self.device = device
        self.output_dir = output_dir
        self.save_name_prefix = save_name_prefix
        self.val_ratio = val_ratio
        self.train_ratio = train_ratio
        self.test_indices_file = test_indices_file
        self.random_seed = random_seed

        # output
        self._total_start = 0.0
        self._total_end = 0.0

    def exec(self, config_file: str) -> None:
        _LOG.info('Start Smart Augment policy generation...')

        ray.init()

        # set random seed
        set_random_seed(self.random_seed)

        # measure duration
        self._total_start = time.time()

        # load configuration file
        smartaug_config = load_config(config_file)

        # create output directory
        os.makedirs(self.output_dir, exist_ok=True)

        # create dataset
        _LOG.info('Dataset: %s', smartaug_config[DATASET].type_key)
        dataset = smartaug_config[DATASET].create_instance()

        # get model config object
        model_config = smartaug_config[MODEL]
        _LOG.info('Model: %s', model_config.type_key)

        # get lables and indices of the data
        labels = normalized_label_ids(dataset)
        indices = list(range(len(labels)))

        _, train_indices = self.test_train_indices(indices, labels)

        # create lists of train labels
        train_labels = [labels[idx] for idx in train_indices]

        # create search space
        space = smart_augment_default_search_space(
            len(self.color_transform_creators()),
            len(self.geometry_transform_creators())
        )

        # split train and validation data
        train_indices, val_indices = split.stratified_train_test_split(
            train_indices,
            train_labels,
            test_size=self.val_ratio,
            random_state=self.random_seed
        )

        train_labels = [labels[idx] for idx in train_indices]

        algo = HyperOptSearch(
            metric='mAP',
            mode='max',
            random_state_seed=self.random_seed
        )

        algo = ConcurrencyLimiter(algo, max_concurrent=self.max_concurrent)

        def trial_dirname_creator(trial: experiment.Trial) -> str:
            # Customize your trial directory name here
            return "smart_aug_trial_" + str(trial.trial_id)

        # for efficient execution on the gcp cluster
        # trainable_with_resources = tune.with_resources(trainable, {"cpu": 4, "gpu": 1})
        trainable_with_resources = trainable

        tuner = tune.Tuner(
            tune.with_parameters(
                trainable_with_resources,
                smart_augment_config_file=os.path.abspath(config_file),
                train_indices=train_indices,
                train_labels=train_labels,
                val_indices=val_indices,
                dataset_path=os.path.abspath(dataset.root),
                epochs=self.epochs,
                batch_size=self.batch_size,
                num_workers=self.num_workers,
                device=self.device,
                random_seed=self.random_seed
            ),
            tune_config=tune.TuneConfig(
                search_alg=algo,
                num_samples=self.num_samples,
                trial_dirname_creator=trial_dirname_creator
            ),
            param_space=space
        )

        _LOG.info('Start tuner ...')
        result_grid = tuner.fit()
        _LOG.info('... tuner finished!')

        # save result datafram to pickle file
        result_df = result_grid.get_dataframe(filter_metric='mAP', filter_mode='max')
        result_df.to_pickle(f'{self.output_dir}/smart_augment_result_df.pkl')
        result_df.to_csv(f'{self.output_dir}/smart_augment_result.csv', index=False)

        # measure duration
        self._total_end = time.time()

        # shut down ray
        ray.shutdown()

        _LOG.info('Took %.3f minutes to find policy.',
                  ((self._total_end - self._total_start) / 60))

    def test_train_indices(
        self,
        indices: List[int],
        labels: List[str]
    ) -> Tuple[List[int], List[int]]:
        return test_train_indices(
            indices,
            labels,
            self.output_dir,
            self.test_indices_file,
            self.train_ratio,
            self.random_seed
        )
