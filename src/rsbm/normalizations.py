"""Data defined as normalized when transformed to unit cube"""

from abc import ABC, abstractmethod

import torch as th


class Normalization(ABC):
    @abstractmethod
    def __call__(self, x: th.Tensor) -> th.Tensor:
        raise NotImplementedError

    @abstractmethod
    def inv(self, x: th.Tensor) -> th.Tensor:
        raise NotImplementedError


class _Identity(Normalization):
    def __call__(self, x: th.Tensor) -> th.Tensor:
        return x

    def inv(self, x: th.Tensor) -> th.Tensor:
        return x


class _PM1(Normalization):
    """Plus-minus one"""

    def __call__(self, x: th.Tensor) -> th.Tensor:
        return (x + 1) / 2

    def inv(self, x: th.Tensor) -> th.Tensor:
        return 2 * x - 1


IDENTITY = _Identity()
PM1 = _PM1()
