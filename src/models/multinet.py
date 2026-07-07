# (D)irectional U-Net
from collections.abc import Hashable

import torch as th
import torch.nn as nn


class MultiNet(nn.Module):
    # Wrapper for multiple UNets
    def __init__(self, module_dict: dict[Hashable, nn.Module]):
        super().__init__()
        self.nets = {}
        for key, module in module_dict.items():
            str_key = f"net_{key}"
            assert str_key not in self.nets, (
                f"Duplicate key in MultiNet when converting to str: {str_key}"
            )
            self.add_module(str_key, module)
            self.nets[key] = module

    def forward(self, x: th.Tensor, t: th.Tensor, direction: int, y=None) -> th.Tensor:
        return self.nets[direction](x, t, y=y)
