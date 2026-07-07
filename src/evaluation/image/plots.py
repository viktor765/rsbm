import os

import numpy as np
import torch as th
from torchvision.utils import make_grid

from ..common import imsave


def plot_images(
    x: th.Tensor, dir: str, label: str, highlight_out_of_bounds: bool = False
) -> None:
    """Saves batch of images. Values out of bounds are either highlighted or clipped."""
    os.makedirs(dir, exist_ok=True)
    channels = x.shape[1]
    if channels not in (1, 3):
        raise ValueError(f"Expected C==1 or C==3, got C={channels}")
    grid = make_grid(x).numpy().transpose(1, 2, 0)
    out_of_bounds = ((grid < 0) | (grid > 1)).any(axis=2)
    if highlight_out_of_bounds and np.any(out_of_bounds):
        grid[out_of_bounds] = (1.0, 0.0, 1.0)
    if not highlight_out_of_bounds:
        grid = np.clip(grid, 0, 1)
    file_name = label + ("_clip" if not highlight_out_of_bounds else "_oob") + ".png"
    imsave(os.path.join(dir, file_name), grid, vmin=0, vmax=1)
