import numpy as np
import torch as th

from ..utils import make_broadcastable
from .reflectors import Reflector


class BrownianBridge:
    # x_t^{0, 1} = (1-t) x0 + t x1 + sigma (B_t - t B_1))
    def __init__(self, sigma):
        self.sigma = sigma

    def sample_given_01(self, x0, x1, t, generator: th.Generator | None = None):
        Z = th.randn(x0.shape, dtype=x0.dtype, device=x0.device, generator=generator)
        t = make_broadcastable(t, x0)
        return (1 - t) * x0 + t * x1 + self.sigma * th.sqrt(t * (1 - t)) * Z

    def score_1_given_t(self, x1, xt, t):
        # gradient is wrt xt
        t = make_broadcastable(t, x1)
        return (x1 - xt) / (self.sigma**2 * (1 - t))

    def score_t_given_0(self, xt, x0, t):
        # gradient is wrt xt
        t = make_broadcastable(t, x0)
        return -(xt - x0) / (self.sigma**2 * t)


class ReflectedBrownianBridge(BrownianBridge):
    def __init__(self, sigma, reflector: Reflector):
        super().__init__(sigma)
        self.reflector = reflector

    def sample_given_01(self, x0, x1, t, generator: th.Generator | None = None):
        batch_size = x0.shape[0]
        signal_shape = x0.shape[1:]
        signal_size = np.prod(signal_shape)
        # n_refls = self.reflector.n_reflections

        # Sample x1_sampled from possible reflections of x1 given x0
        # Since density factorizes, we sample each dimension independently
        x1 = x1.reshape(-1)  # (1, batch_size * signal_size)
        x1_refls = self.reflector.get_reflections(
            x1
        )  # possible unreflected endpoints, (n_refls, batch_size * signal_size)
        x0 = x0.reshape(-1).unsqueeze(0)  # (1, batch_size * signal_size)

        probs = th.softmax(
            -((x1_refls - x0) ** 2) / (2 * self.sigma**2 * (1 - 0)), dim=0
        )  # (n_refls, batch_size * signal_size)
        indices = th.multinomial(
            probs.mT, 1, generator=generator, replacement=True
        ).squeeze(-1)  # (batch_size * signal_size,)
        # replacement=True not really
        # needed since num_samples=1
        x1_sampled = x1_refls[
            indices, th.arange(batch_size * signal_size, device=x0.device)
        ]  # (batch_size * signal_size,)
        x1_sampled = x1_sampled.reshape(
            batch_size, *signal_shape
        )  # (batch_size, *signal_shape)
        x0 = x0.squeeze(0).reshape(
            batch_size, *signal_shape
        )  # (batch_size, *signal_shape)

        # Sample from bridge given x0 and x1_sampled
        Z = th.randn(x0.shape, dtype=x0.dtype, device=x0.device, generator=generator)
        t = make_broadcastable(t, x0)
        return self.reflector.reflect(
            (1 - t) * x0 + t * x1_sampled + self.sigma * th.sqrt(t * (1 - t)) * Z
        )

    def score_1_given_t(self, x1, xt, t):
        # gradient is wrt xt
        t = make_broadcastable(t, x1).unsqueeze(0)
        x1_refls = self.reflector.get_reflections(x1)
        xt = xt.unsqueeze(0)
        exponents = -((x1_refls - xt) ** 2) / (2 * self.sigma**2 * (1 - t))
        dexponents_dxt = (x1_refls - xt) / (self.sigma**2 * (1 - t))
        weights = th.exp(
            exponents - th.max(exponents, dim=0, keepdim=True).values
        )  # for numerical stability
        score = th.sum(dexponents_dxt * weights, dim=0) / th.sum(weights, dim=0)
        return score

    # # Alternative form
    # def _alt_score_1_given_t(self, x1, xt, t):
    #     t = make_broadcastable(t, x1).unsqueeze(0)
    #     x1 = x1.unsqueeze(0)
    #     xt_refls = self.reflector.get_reflections(xt)
    #     exponents = -(x1 - xt_refls) ** 2 / (2 * self.sigma ** 2 * (1 - t))
    #     signs = make_broadcastable(self.reflector.drefl_dx(device=x1.device), xt_refls)
    #     dexponents_dxt = signs * (x1 - xt_refls) / (self.sigma ** 2 * (1 - t))
    #     weights = th.exp(exponents - th.max(exponents, dim=0, keepdim=True).values)
    #     score = th.sum(dexponents_dxt * weights, dim=0) / th.sum(weights, dim=0)
    #     return score

    def score_t_given_0(self, xt, x0, t):
        # gradient is wrt xt
        t = make_broadcastable(t, x0).unsqueeze(0)
        x0_refls = self.reflector.get_reflections(x0)
        xt = xt.unsqueeze(0)
        exponents = -((xt - x0_refls) ** 2) / (2 * self.sigma**2 * t)
        dexponents_dxt = -(xt - x0_refls) / (self.sigma**2 * t)
        weights = th.exp(
            exponents - th.max(exponents, dim=0, keepdim=True).values
        )  # for numerical stability
        score = th.sum(dexponents_dxt * weights, dim=0) / th.sum(weights, dim=0)
        return score

    # # Alternative form
    # def _alt_score_t_given_0(self, xt, x0, t):
    #     t = make_broadcastable(t, x0).unsqueeze(0)
    #     x0 = x0.unsqueeze(0)
    #     xt_refls = self.reflector.get_reflections(xt)
    #     exponents = -(xt_refls - x0) ** 2 / (2 * self.sigma ** 2 * t)
    #     signs = make_broadcastable(self.reflector.drefl_dx(device=x0.device), xt_refls)
    #     dexponents_dxt = -signs * (xt_refls - x0) / (self.sigma ** 2 * t)
    #     weights = th.exp(exponents - th.max(exponents, dim=0, keepdim=True).values)
    #     score = th.sum(dexponents_dxt * weights, dim=0) / th.sum(weights, dim=0)
    #     return score
