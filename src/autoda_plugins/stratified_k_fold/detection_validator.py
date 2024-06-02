from typing import Any

import torch

from torch import nn
from torch.utils import data as torch_data
from torch._prims_common import DeviceLikeType

from tqdm import tqdm


class DetectionValidator:
    def run(
        self,
        *,
        model: nn.Module,
        dataloader: torch_data.DataLoader[torch_data.Dataset[Any]],
        device: DeviceLikeType = "cpu",
        progress: bool = True
    ) -> Any:
        model.eval()

        # initialize tqdm progress bar
        prog_bar = tqdm(dataloader, total=len(dataloader), disable=not progress)

        target_list = []
        preds_list = []

        for data, targets in prog_bar:

            data = [d.to(device) for d in data]
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

            with torch.no_grad():
                outputs = model(data)

            post_targets = []
            post_preds = []

            for i in range(len(targets)):
                true_dict = {}
                preds_dict = {}
                true_dict['boxes'] = targets[i]['boxes'].detach().cpu()
                true_dict['labels'] = targets[i]['labels'].detach().cpu()
                preds_dict['boxes'] = outputs[i]['boxes'].detach().cpu()
                preds_dict['scores'] = outputs[i]['scores'].detach().cpu()
                preds_dict['labels'] = outputs[i]['labels'].detach().cpu()
                post_targets.append(true_dict)
                post_preds.append(preds_dict)

            target_list.extend(post_targets)
            preds_list.extend(post_preds)

        return target_list, preds_list
