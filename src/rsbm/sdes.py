from contextlib import nullcontext

import torch as th

from .reflectors import Reflector


class SDE:
    def __init__(self, sigma: float, euler_steps: int):
        self.sigma = sigma
        self.euler_steps = euler_steps

    @th.no_grad()
    def forward_euler(
        self,
        model,
        x0: th.Tensor,
        direction: int,
        generator: th.Generator | None = None,
        autocast=None,
        return_trajectory: bool = False,
    ) -> th.Tensor | tuple[th.Tensor, th.Tensor]:
        autocast_context = nullcontext if autocast is None else autocast
        x = x0
        trajectory = [x] if return_trajectory else None
        ts = th.linspace(0, 1, self.euler_steps + 1, device=x0.device, dtype=x0.dtype)
        for i in range(self.euler_steps):
            t = ts[i].repeat(x.shape[0])
            dt = ts[i + 1] - ts[i]
            with autocast_context():
                score = model(x, t, direction=direction)
            x = x + score.to(x.dtype) * dt
            is_last_step = i == self.euler_steps - 1
            if not is_last_step:
                noise = th.randn(
                    x.shape, dtype=x.dtype, device=x.device, generator=generator
                )
                x = x + self.sigma * noise * th.sqrt(dt)
            x = self._post_step(x, args={"is_last_step": is_last_step})
            if trajectory is not None:
                trajectory.append(x)
        if trajectory is not None:
            return x, th.stack(trajectory, dim=1)
        return x

    def _post_step(self, x: th.Tensor, args: dict | None = None) -> th.Tensor:
        return x


class ReflectedSDE(SDE):
    def __init__(self, sigma: float, euler_steps: int, reflector: Reflector):
        super().__init__(sigma, euler_steps)
        self.reflector = reflector

    def _post_step(self, x: th.Tensor, args: dict | None = None) -> th.Tensor:
        if args is not None and args.get("is_last_step", False):
            return self.reflector.project(x)

        return self.reflector.reflect(x)
