import torch as th

from .directions import DirectionSpec


def sample_direction(
    *,
    sde,
    model: th.nn.Module,
    source: th.Tensor,
    direction: DirectionSpec,
    device: th.device,
    generator: th.Generator,
    autocast,
    return_trajectory: bool = False,
):
    sampled = sde.forward_euler(
        model,
        source.to(device),
        direction=direction.sampling_direction,
        generator=generator,
        autocast=autocast,
        return_trajectory=return_trajectory,
    )
    if return_trajectory:
        final, trajectory = sampled
        return final.cpu(), trajectory.cpu()
    return sampled.cpu()
