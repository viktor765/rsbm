import torch as th

from ..utils import make_broadcastable
from .normalizations import Normalization


class Reflector:
    def __init__(self, n_reflections: int, normalization: Normalization):
        if n_reflections < 1:
            raise ValueError(f"n_reflections must be >= 1, got {n_reflections}")
        self.n_reflections = n_reflections
        self.normalization = normalization

    def reflect(self, x: th.Tensor) -> th.Tensor:
        """
        Reflect the input tensor x into the unit cube [0, 1]^d.
        """
        x = self.normalization(x)
        x = x % 2
        over_one = x > 1
        x[over_one] = 2 - x[over_one]
        return self.normalization.inv(x)

    def project(self, x: th.Tensor) -> th.Tensor:
        """
        Project the input tensor x into the unit cube [0, 1]^d.
        """
        x = self.normalization(x)
        x = th.clamp(x, 0.0, 1.0)
        return self.normalization.inv(x)

    def drefl_dx(self, device: th.device | None = None) -> th.Tensor:
        """
        Compute the derivative of the reflection function with respect to x.
        """
        n = self.n_reflections
        rng = th.arange(n, device=device) + ((n // 2) % 2)
        return 1 - 2 * (rng % 2)

    def get_reflections(self, x: th.Tensor) -> th.Tensor:
        """
        Generate reflected versions of x. Output shape: (num_reflections, *x.shape).

        Example n=7:
        -2 - x
        -2 + x
         0 - x
         0 + x
         2 - x
         2 + x
         4 - x
        """
        n = self.n_reflections
        rng = th.arange(n, device=x.device) - n // 2 + 1
        offsets = rng - (rng % 2)

        x = x.unsqueeze(0)
        offsets = make_broadcastable(offsets, x)
        signs = make_broadcastable(self.drefl_dx(x.device), x)

        reflections = self.normalization.inv(offsets + signs * self.normalization(x))
        return reflections
