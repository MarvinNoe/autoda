import copy
import json
import logging
import os
import time

from filelock import FileLock
from pathlib import Path
from typing import Dict, Optional, List, Any, Tuple, Callable

import pandas as pd
import ray
import torch

from ray import tune
from ray.tune.search.hyperopt import HyperOptSearch
from ray.tune.search import ConcurrencyLimiter
from torch import nn
from torch.utils import data
from torchmetrics.detection import mean_ap
from torchvision.transforms import v2

import autoda.globals as autoda_globals

from autoda.abstract import Routine
from autoda.config.config_object import ConfigObject
from autoda.config.yaml_loader import load_config, load_plugins

from .policy import AutoDaPolicy
from .policy_decode import decode_fastaa_policy_config
from .space import fastaa_search_space_hyperopt
from ..plugins.globals import TRANSF_CREATOR_COLL_NAME
from ..stratified_k_fold.stratified_k_fold import collate_fn, stratified_k_fold_cross_validation
from ..stratified_k_fold.detection_validator import DetectionValidator
from ..utils.random_setup import set_random_seed
from ..utils.torch.indices import test_train_indices, indices_from_file
from ..utils.torch.labels import normalized_label_ids

_LOG: logging.Logger = logging.getLogger(__name__)

MODEL = autoda_globals.MODEL_FACTORY_NAME
DATASET = autoda_globals.DATASET_FACTORY_NAME
ROUTINE = autoda_globals.ROUTINE_FACTORY_NAME
TRANSFORMS = TRANSF_CREATOR_COLL_NAME


def create_model(model_config: ConfigObject, checkpoint_path: Optional[Path] = None) -> nn.Module:
    model_config = copy.deepcopy(model_config)
    model = model_config.create_instance()

    if checkpoint_path is not None:
        checkpoint = torch.load(checkpoint_path, weights_only=True)
        model.load_state_dict(checkpoint['model_state_dict'])

    return model


def create_augmented_dataset(
    dataset_config: ConfigObject,
    transforms_creator_config: ConfigObject,
    augment_policy: AutoDaPolicy
) -> data.Dataset:
    dataset_config_copy = copy.deepcopy(dataset_config)

    transforms_creator = transforms_creator_config.create_instance()
    transform_policy = augment_policy.torchvision_random_choice_v2(transforms_creator())

    try:
        transforms_config = dataset_config_copy.get_arg('transforms')
    except KeyError:
        transforms_config = None

    if transforms_config is not None:
        default_transforms = transforms_config.create_instance()
        transform_policy = v2.Compose([default_transforms, transform_policy])

    dataset_config_copy.set_arg('transforms', transform_policy)

    with FileLock(os.path.expanduser("~/.data.lock")):
        dataset = dataset_config_copy.create_instance()

    return dataset


def trainable(
    config: Dict[str, Any],
    autoda_config_file: str,
    indices=None,
    model_path=None,
    dataset_path=None,
    num_policies=None,
    num_ops=None,
    batch_size=None,
    num_workers=None,
    device=None,
    random_seed=None
):
    # load plugins and configurations from file
    try:
        load_plugins(autoda_config_file)
    except KeyError:
        _LOG.info("Plugins already laoded.")

    autoda_config = load_config(autoda_config_file)

    # create model
    model = create_model(autoda_config[MODEL], model_path)
    model.to(device)
    model.eval()

    # get augmentations from config
    policy = decode_fastaa_policy_config(config, num_policies, num_ops)

    # create dataset
    autoda_config[DATASET].set_arg('root', dataset_path)
    dataset = create_augmented_dataset(
        autoda_config[DATASET],
        autoda_config[ROUTINE].get_arg(TRANSFORMS),
        policy
    )

    # create dataloader
    aug_loader = data.DataLoader(
        data.Subset(dataset, indices),
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=collate_fn,
        drop_last=True,
        pin_memory=True
    )

    # validate augmented data
    validator = DetectionValidator()

    target, pred = validator.run(
        model=model,
        dataloader=aug_loader,
        device=device,
        progress=True
    )

    metric = mean_ap.MeanAveragePrecision(class_metrics=False)
    metric.update(pred, target)
    metric_summary = metric.compute()

    return {'mAP': metric_summary['map'].item()}


class FastAaRoutine(Routine):
    def __init__(
        self,
        epochs: int,
        batch_size: int,
        num_folds: int,
        num_steps: int,
        num_samples: int,
        top_n: int,
        num_workers: int,
        num_policies: int,
        num_ops: int,
        max_concurrent: int,
        transforms_creator: Callable[..., Any],
        device: str,
        output_dir: str,
        save_name_prefix: str,
        train_ratio: Optional[int] = None,
        test_indices_file: Optional[str] = None,
        random_seed: Optional[int] = None
    ):
        if (train_ratio is None and test_indices_file is None) or \
                (train_ratio is not None and test_indices_file is not None):
            raise ValueError("Exactly one of train_ratio or test_indices_file must be provided.")

        # input
        self.epochs = epochs
        self.batch_size = batch_size
        self.num_folds = num_folds
        self.num_steps = num_steps
        self.num_workers = num_workers
        self.num_samples = num_samples
        self.top_n = top_n
        self.num_policies = num_policies
        self.num_ops = num_ops
        self.max_concurrent = max_concurrent
        self.transforms_creator = transforms_creator
        self.device = device
        self.output_dir = output_dir
        self.save_name_prefix = save_name_prefix
        self.train_ratio = train_ratio
        self.test_indices_file = test_indices_file
        self.random_seed = random_seed

        # output
        self._final_policies_top_n = pd.DataFrame()
        self._total_start = 0.0
        self._total_end = 0.0

    def exec(self, config_file: str) -> None:
        _LOG.info('Start FastAA policy generation...')

        # init ray
        ray.init()

        # set random seed
        set_random_seed(self.random_seed)

        # measure duration
        self._total_start = time.time()

        # load configuration file
        autoda_config = load_config(config_file)

        # create output directory
        os.makedirs(self.output_dir, exist_ok=True)

        # create dataset
        _LOG.info('Dataset: %s', autoda_config[DATASET].type_key)
        dataset = autoda_config[DATASET].create_instance()

        # get model config object
        model_config = autoda_config[MODEL]
        _LOG.info('Model: %s', model_config.type_key)

        # get lables and indices of the data
        labels = normalized_label_ids(dataset)
        indices = list(range(len(labels)))

        _, train_indices = self.test_train_indices(indices, labels)

        # create lists of train labels
        train_labels = [labels[idx] for idx in train_indices]

        # get augmentation indices file paths
        aug_indices_file_prefix = 'aug_indices'
        aug_indices_paths = [
            str((Path(self.output_dir) / f'{aug_indices_file_prefix}_fold_{i}.txt').resolve())
            for i in range(self.num_folds)
        ]

        # check whether model parameter files for all folds already exist in the output directory
        model_paths = [
            (Path(self.output_dir) / f'{self.save_name_prefix}_fold_{i}.pth').resolve()
            for i in range(self.num_folds)
        ]

        pretrained_models_exist = all(os.path.isfile(model_paths[i]) for i in range(self.num_folds))

        if not pretrained_models_exist:
            # train the models
            _LOG.info("Train models...")
            training_start = time.time()

            stratified_k_fold_cross_validation(
                folds=self.num_folds,
                train_indices=train_indices,
                train_labels=train_labels,
                config=autoda_config,
                epochs=self.epochs,
                batch_size=self.batch_size,
                num_workers=self.num_workers,
                device=self.device,
                output_dir=self.output_dir,
                random_seed=self.random_seed,
                val_indices_file_prefix=aug_indices_file_prefix
            )
            trianing_end = time.time()
            _LOG.info("...models trained!")
            _LOG.info('Took %.3f minutes to train models.',
                      ((trianing_end - training_start) / 60))

        # create search space
        space = fastaa_search_space_hyperopt(
            self.num_policies,
            self.num_ops,
            len(self.transforms_creator())
        )

        # start FastAA algorithm
        for fold in range(self.num_folds):  # K
            _LOG.info("Fold: %i", fold)
            aug_indices = indices_from_file(aug_indices_paths[fold])
            model_path = model_paths[fold]

            for t in range(self.num_steps):  # T
                _LOG.info("t: %i", t)

                algo = HyperOptSearch(
                    space=space,
                    metric='mAP',
                    mode='max',
                    random_state_seed=self.random_seed
                )

                algo = ConcurrencyLimiter(algo, max_concurrent=self.max_concurrent)  # paper

                tuner = tune.Tuner(
                    tune.with_parameters(
                        trainable,
                        autoda_config_file=os.path.abspath(config_file),
                        indices=aug_indices,
                        model_path=model_path,
                        dataset_path=os.path.abspath(dataset.root),
                        num_policies=self.num_policies,
                        num_ops=self.num_ops,
                        batch_size=self.batch_size,
                        num_workers=self.num_workers,
                        device=self.device,
                        random_seed=self.random_seed
                    ),
                    tune_config=tune.TuneConfig(
                        search_alg=algo,
                        num_samples=self.num_samples  # B
                    )
                )

                _LOG.info('Start tuner ...')
                result_grid = tuner.fit()
                _LOG.info('... tuner finished!')

                self.save_results(fold, t, result_grid)

        # measure duration
        self._total_end = time.time()

        # shut down ray
        ray.shutdown()

        # create output file
        self.create_output_files()

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

    def save_results(self, fold: int, t: int, result_grid: ray.tune.ResultGrid) -> None:
        _LOG.info('Save results...')

        # save result datafram to pickle file
        result_df = result_grid.get_dataframe(filter_metric='mAP', filter_mode='max')
        result_df.to_pickle(f'{self.output_dir}/policies_fold_{fold}_t_{t}_df.pkl')

        # save top-n policies
        self._final_policies_top_n = pd.concat(
            [
                self._final_policies_top_n,
                result_df.nlargest(self.top_n, 'mAP')
            ]
        )

        _LOG.info('...results saved!')

    def create_output_files(self) -> None:
        top_n_df = self._final_policies_top_n

        # save finale policies to a CSV file
        top_n_df.to_csv(f'{self.output_dir}/policies_top_{self.top_n}.csv', index=False)

        top_1_df = top_n_df.iloc[::self.top_n]
        top_1_df.to_csv(f'{self.output_dir}/policies_top_1.csv', index=False)

        top_2_df = pd.concat([top_n_df.iloc[i:i+2] for i in range(0, len(top_n_df), self.top_n)])
        top_2_df.to_csv(f'{self.output_dir}/policies_top_2.csv', index=False)

        best_policy_df = top_n_df.nlargest(1, 'mAP')
        best_policy_df.to_csv(f'{self.output_dir}/policy_best.csv', index=False)

        # serialize timestamps
        with open(f'{self.output_dir}/timestamps.json', 'w') as f:
            json.dump({'total_start': self._total_start, 'total_end': self._total_end}, f)
