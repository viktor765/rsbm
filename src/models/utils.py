import copy

import torch as th
import torch.nn as nn


def create_eval_copy(model: nn.Module) -> nn.Module:
    model_copy = copy.deepcopy(model)
    model_copy.requires_grad_(False)
    model_copy.eval()
    return model_copy


def update_ema(base_model: nn.Module, ema_model: nn.Module, ema_rate: float) -> None:
    for param, ema_param in zip(
        base_model.parameters(), ema_model.parameters(), strict=True
    ):
        ema_param.data.mul_(ema_rate).add_(param, alpha=1 - ema_rate)
    buffers = dict(base_model.named_buffers())
    for name, ema_buffer in ema_model.named_buffers():
        ema_buffer.data.copy_(buffers[name])


def direction_to_01(direction, batch_size: int, device: th.device) -> th.Tensor:
    # Replicate convention used in De Bortoli et al. (2024) "alpha-IMF"
    if direction == 1:
        return th.ones(batch_size, device=device, dtype=th.int64)
    elif direction == -1:
        return th.zeros(batch_size, device=device, dtype=th.int64)
    else:
        raise ValueError("Direction must be either +1 or -1.")
