from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

import torch
from torch import Tensor

from models import EncoderSize, Net


def plot_losses(train_loss, n_train, val_loss, n_val, save_dir):
    train_loss = np.array(train_loss).reshape(-1, n_train)
    val_loss = np.array(val_loss).reshape(-1, n_val)
    fig, ax = plt.subplots(1, 1, figsize=(8, 5))
    x = np.arange(1, train_loss.shape[0] + 1)
    ax.plot(
        x,
        train_loss.mean(axis=1),
        color="dodgerblue", label="train"
    )
    ax.plot(
        x,
        val_loss.mean(axis=1),
        color="coral", label="validation"
    )
    ax.grid(alpha=0.35)
    ax.legend(framealpha=0.6)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.set_title("Training and Validation Losses")
    fig.savefig(save_dir / "loss.png")
    # plt.show()


def load_encoder_weights(model: Net, encoder_size: EncoderSize, device):
    base_dir = Path(__file__).parent
    weight_path = None
    if encoder_size == EncoderSize.SMALL:
        print("using Hiera-Small encoder.")
        weight_path = base_dir / "hiera_small_encoder.pt"
    elif encoder_size == EncoderSize.TINY:
        print("using Hiera-Tiny encoder.")
        weight_path = base_dir / "hiera_tiny_encoder.pt"
    elif encoder_size == EncoderSize.BASE:
        print("using Hiera-Base+ encoder.")
        weight_path = base_dir / "hiera_base+_encoder.pt"
    elif encoder_size == EncoderSize.LARGE:
        print("using Hiera-Large encoder.")
        weight_path = base_dir / "hiera_large_encoder.pt"
    else:
        raise ValueError("encoder size must be one of [tiny, small, base+, large].")

    if weight_path.exists():
        model.load_encoder_weights(
            torch.load(weight_path, map_location=device)
        )
    else:
        raise FileNotFoundError(f"encoder weights not found: {weight_path}")


def dice_coeff(
    input: Tensor, target: Tensor,
    reduce_batch_first: bool = False, epsilon: float = 1e-6
):
    # Average of Dice coefficient for all batches, or for a single mask
    assert input.size() == target.size()
    assert input.dim() == 4 or not reduce_batch_first

    sum_dim = (-1, -2) if input.dim() == 2 or not reduce_batch_first else (-1, -2, -3)

    inter = 2 * (input * target).sum(dim=sum_dim)
    sets_sum = input.sum(dim=sum_dim) + target.sum(dim=sum_dim)
    sets_sum = torch.where(sets_sum == 0, inter, sets_sum)

    dice = (inter + epsilon) / (sets_sum + epsilon)
    return dice.mean()


def dice_loss(input: Tensor, target: Tensor, is_prob: bool = False):
    # Dice loss (objective to minimize) between 0 and 1
    if is_prob:
        probs = input
    else:
        probs = input.sigmoid()
    return 1 - dice_coeff(probs, target, reduce_batch_first=True)
