import random

import numpy as np
import torch as th


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    th.manual_seed(seed)
    th.cuda.manual_seed(seed)


def create_generators(seed: int, device: th.device | str) -> th.Generator:
    gen = th.Generator(device)
    gen.manual_seed(seed)
    return gen


def make_broadcastable(tensor: th.Tensor, other: th.Tensor) -> th.Tensor:
    return tensor.view(*tensor.shape, *([1] * (other.dim() - tensor.dim())))
