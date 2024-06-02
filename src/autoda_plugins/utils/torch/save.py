from typing import List

import torch
import matplotlib.pyplot as plt

from torch import nn
from torch.optim import Optimizer


class SaveBestModel:
    """
    Class to save the best model while training. If the current epoch's
    validation mAP @0.5:0.95 IoU higher than the previous highest, then save the
    model state.
    """

    def __init__(self, best_valid_map: float = 0.):
        self.best_valid_map = best_valid_map

    def __call__(
        self,
        model: nn.Module,
        current_valid_map: float,
        epoch: int,
        out_dir: str,
        save_name: str
    ) -> None:
        if current_valid_map > self.best_valid_map:
            self.best_valid_map = current_valid_map
            print(f"\nBEST VALIDATION mAP: {self.best_valid_map}")
            print(f"\nSAVING BEST MODEL FOR EPOCH: {epoch+1}\n")
            torch.save({
                'epoch': epoch+1,
                'model_state_dict': model.state_dict(),
            }, f"{out_dir}/{save_name}.pth")


def save_model(epoch: int, model: nn.Module, optimizer: Optimizer) -> None:
    """
    Function to save the trained model till current epoch, or whenver called
    """
    torch.save({
        'epoch': epoch+1,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
    }, 'outputs/last_model.pth')


def save_loss_plot(
    out_dir: str,
    train_loss_list: List[float],
    x_label: str = 'iterations',
    y_label: str = 'train loss',
    save_name: str = 'train_loss'
) -> None:
    """
    Function to save both train loss graph.

    :param out_dir: Path to save the graphs.
    :param train_loss_list: List containing the training loss values.
    """
    figure_1 = plt.figure(figsize=(10, 7), num=1, clear=True)
    train_ax = figure_1.add_subplot()
    train_ax.plot(train_loss_list, color='tab:blue')
    train_ax.set_xlabel(x_label)
    train_ax.set_ylabel(y_label)
    figure_1.savefig(f"{out_dir}/{save_name}.png")
    print('SAVING PLOTS COMPLETE...')


def save_m_ap(
    out_dir: str,
    map_05: List[float],
    map: List[float],
    save_name: str = 'map'
) -> None:
    """
    Saves the mAP@0.5 and mAP@0.5:0.95 per epoch.
    :param out_dir: Path to save the graphs.
    :param map_05: List containing mAP values at 0.5 IoU.
    :param map: List containing mAP values at 0.5:0.95 IoU.
    """
    figure = plt.figure(figsize=(10, 7), num=1, clear=True)
    ax = figure.add_subplot()
    ax.plot(
        map_05, color='tab:orange', linestyle='-',
        label='mAP@0.5'
    )
    ax.plot(
        map, color='tab:red', linestyle='-',
        label='mAP@0.5:0.95'
    )
    ax.set_xlabel('Epochs')
    ax.set_ylabel('mAP')
    ax.legend()
    figure.savefig(f"{out_dir}/{save_name}.png")
