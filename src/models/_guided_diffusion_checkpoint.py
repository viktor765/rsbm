import torch as th

from ..guided_diffusion import nn as guided_diffusion_nn
from ..guided_diffusion import unet as guided_diffusion_unet


def _autocast_device_type(inputs) -> str:
    for x in inputs:
        if isinstance(x, th.Tensor):
            return x.device.type
    return "cuda" if th.cuda.is_available() else "cpu"


def _is_autocast_enabled(device_type: str) -> bool:
    try:
        return th.is_autocast_enabled(device_type)
    except TypeError:
        if device_type == "cpu" and hasattr(th, "is_autocast_cpu_enabled"):
            return th.is_autocast_cpu_enabled()
        return th.is_autocast_enabled()


def _get_autocast_dtype(device_type: str) -> th.dtype:
    try:
        return th.get_autocast_dtype(device_type)
    except TypeError:
        if device_type == "cuda" and hasattr(th, "get_autocast_gpu_dtype"):
            return th.get_autocast_gpu_dtype()
        if device_type == "cpu" and hasattr(th, "get_autocast_cpu_dtype"):
            return th.get_autocast_cpu_dtype()
        return th.float16 if device_type == "cuda" else th.bfloat16


class _AutocastCheckpointFunction(th.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        run_function,
        length,
        device_type,
        autocast_enabled,
        autocast_dtype,
        cache_enabled,
        *args,
    ):
        ctx.run_function = run_function
        ctx.input_tensors = list(args[:length])
        ctx.input_params = list(args[length:])
        ctx.device_type = device_type
        ctx.autocast_enabled = autocast_enabled
        ctx.autocast_dtype = autocast_dtype
        ctx.cache_enabled = cache_enabled
        with th.no_grad():
            output_tensors = ctx.run_function(*ctx.input_tensors)
        return output_tensors

    @staticmethod
    def backward(ctx, *output_grads):
        ctx.input_tensors = [x.detach().requires_grad_(True) for x in ctx.input_tensors]
        with th.enable_grad():
            shallow_copies = [x.view_as(x) for x in ctx.input_tensors]
            with th.amp.autocast(
                device_type=ctx.device_type,
                enabled=ctx.autocast_enabled,
                dtype=ctx.autocast_dtype,
                cache_enabled=ctx.cache_enabled,
            ):
                output_tensors = ctx.run_function(*shallow_copies)
        input_grads = th.autograd.grad(
            output_tensors,
            ctx.input_tensors + ctx.input_params,
            output_grads,
            allow_unused=True,
        )
        del ctx.input_tensors
        del ctx.input_params
        del output_tensors
        return (None, None, None, None, None, None) + input_grads


def checkpoint(func, inputs, params, flag):
    if not flag:
        return func(*inputs)

    device_type = _autocast_device_type(inputs)
    args = tuple(inputs) + tuple(params)
    return _AutocastCheckpointFunction.apply(
        func,
        len(inputs),
        device_type,
        _is_autocast_enabled(device_type),
        _get_autocast_dtype(device_type),
        th.is_autocast_cache_enabled(),
        *args,
    )


def install() -> None:
    guided_diffusion_nn.checkpoint = checkpoint
    guided_diffusion_unet.checkpoint = checkpoint

