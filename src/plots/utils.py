import os
import yaml

import pandas as pd


def read_map_values(base_dir: str):
    data = []
    experiments = sorted(os.listdir(base_dir))

    for exp in experiments:
        exp_dir = os.path.join(base_dir, exp)
        if os.path.isdir(exp_dir):
            fold_files = sorted(os.listdir(exp_dir))
            for fold_file in fold_files:
                if fold_file.startswith("metric_fold_") and fold_file.endswith(".yaml"):
                    fold_id = int(fold_file.split('_')[2].split('.')[0])
                    file_path = os.path.join(exp_dir, fold_file)
                    with open(file_path, 'r') as file:
                        metrics = yaml.safe_load(file)
                        map_value = metrics.get('map')
                        data.append({
                            'Experiment': exp,
                            'Fold': fold_id,
                            'mAP': map_value
                        })
    return data
