import copy
import glob
import logging
import os
import shutil
import time
import yaml

from typing import Union, List, Optional, Tuple, Any, Dict

import torch

from torch.utils import data
from torchmetrics.detection import mean_ap

from autoda.abstract import Routine
from autoda.config.yaml_loader import load_config

from .detection_validator import DetectionValidator
from .stratified_k_fold import collate_fn
from ..utils.torch import split
from ..utils.torch.indices import indices_from_file, indices_to_file
from ..utils.torch.labels import normalized_label_ids

_LOG = logging.getLogger(__name__)


class ValidatedKFoldRoutine(Routine):
    def __init__(
        self,
        param_files: Union[str, List[str]],
        output_dir: str,
        batch_size: int,
        num_workers: int,
        device: str,
        test_ratio: Optional[int] = None,
        test_indices_file: Optional[str] = None,
        random_seed: Optional[int] = None
    ):
        if (test_ratio is None and test_indices_file is None) or \
                (test_ratio is not None and test_indices_file is not None):
            raise ValueError("Exactly one of test_ratio or test_indices_file must be provided.")

        self.param_files = param_files

        if isinstance(self.param_files, str):
            self.param_files = glob.glob(self.param_files)

        self.output_dir = output_dir
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.device = torch.device(device)
        self.test_ratio = test_ratio
        self.test_indices_file = test_indices_file
        self.random_seed = random_seed

        # output
        self.accumulated_metric: Dict[str, float] = {}

    def exec(self, config_file: str) -> None:
        _LOG.info("Execute ValidateKFold...")

        total_start = time.time()

        # load configuration file
        config = load_config(config_file)

        # create output directory
        os.makedirs(self.output_dir, exist_ok=True)

        # create dataset
        dataset = config['dataset'].create_instance()

        # get test indices
        test_indices = self.test_indices(dataset)

        # create dataloader
        val_loader = data.DataLoader(
            data.Subset(dataset, test_indices),
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            collate_fn=collate_fn,
            drop_last=True,
            pin_memory=True
        )

        # create validator
        validator = DetectionValidator()

        # validate folds
        for fold, file in enumerate(self.param_files):
            _LOG.info('[Fold: %s, File: %s] Start validation...', fold, file)

            # create model and load parameters
            model = config['model'].create_instance()
            checkpoint = torch.load(file, weights_only=True)
            model.load_state_dict(checkpoint['model_state_dict'])
            model.to(self.device)

            target, pred = validator.run(
                model=model,
                dataloader=val_loader,
                device=self.device,
                progress=True
            )

            metric = mean_ap.MeanAveragePrecision(class_metrics=True)
            metric.update(pred, target)
            metric_summary = metric.compute()

            self.metric_to_yaml_file(f'metric_fold_{fold}', metric_summary)
            self.accumulate_metric(metric_summary)

            _LOG.info('[Fold: %s, File: %s] mAP %s', fold, file, metric_summary['map'].item())

        # calculate final metric
        final_metric = self.calculate_final_metric(len(self.param_files))
        self.metric_to_yaml_file('final_metric', final_metric)

        # validation ends
        total_end = time.time()
        print("Final metric: ", final_metric)
        _LOG.info('Total training time: %.3f minutes.', ((total_end - total_start) / 60))

    def test_indices(self, dataset: data.Dataset[Tuple[Any, Any]]) -> List[int]:
        test_indices = []

        if self.test_indices_file is not None:
            # copy test indices into output dir
            shutil.copyfile(self.test_indices_file, f'{self.output_dir}/test_indices.txt')

            # load test indices from file
            test_indices = indices_from_file(f'{self.output_dir}/test_indices.txt')

        elif self.test_ratio is not None:
            train_ratio = 1.0 - self.test_ratio

            # get labels and indices of the data
            labels = normalized_label_ids(dataset)
            indices = list(range(len(labels)))

            # split the list of indices into train and test
            _, test_indices = split.stratified_train_test_split(
                indices,
                labels,
                train_size=train_ratio,
                random_state=self.random_seed
            )

            indices_to_file(test_indices, f'{self.output_dir}/test_indices.txt')

        return test_indices

    def metric_to_yaml_file(self, file_name: str, metric_dict: Dict[str, Any]) -> None:
        metric = copy.deepcopy(metric_dict)

        # Convert tensor values to numpy arrays
        for key, value in metric.items():
            if isinstance(value, torch.Tensor):
                metric[key] = value.numpy().tolist()

        # write metric to json file
        with open(f'{self.output_dir}/{file_name}.yaml', 'w') as json_file:
            yaml.dump(metric, json_file)

    def accumulate_metric(self, metric_dict: Dict[str, Any]) -> None:
        for key in metric_dict.keys():
            if key == 'classes':
                continue

            if isinstance(metric_dict[key], torch.Tensor):
                if key in self.accumulated_metric:
                    self.accumulated_metric[key] += metric_dict[key]
                else:
                    self.accumulated_metric[key] = metric_dict[key]

    def calculate_final_metric(self, divisor: int) -> Dict[str, Any]:
        final_metric = self.accumulated_metric

        for key in final_metric.keys():
            if key == 'classes':
                continue
            if isinstance(final_metric[key], torch.Tensor):
                final_metric[key] = final_metric[key] / divisor

        return final_metric
