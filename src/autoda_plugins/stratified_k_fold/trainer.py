from typing import Any, Optional

import torch.optim as optim

from torch import nn
from torch.utils import data as torch_data
from torch._prims_common import DeviceLikeType

from tqdm import tqdm


class Trainer:
    def run(
        self,
        *,
        model: nn.Module,
        dataloader: torch_data.DataLoader[torch_data.Dataset[Any]],
        optimizer: optim.Optimizer,
        device: DeviceLikeType = 'cpu',
        progress: Optional[bool] = True
    ) -> Any:
        model.train()
        model.to(device)

        # initialize tqdm progress bar
        prog_bar = tqdm(dataloader, total=len(dataloader), disable=not progress)

        total_loss = 0.

        for data, targets in prog_bar:
            optimizer.zero_grad()

            data = [d.to(device) for d in data]
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

            loss_dict = model(data, targets)
            losses = sum(loss for loss in loss_dict.values())

            loss_value = losses.item()
            total_loss += loss_value

            losses.backward()
            optimizer.step()

            # update progess bar
            prog_bar.set_description(desc=f'Loss: {loss_value:.4f}')

        return loss_value, total_loss / len(dataloader)
