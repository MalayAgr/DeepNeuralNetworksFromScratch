from __future__ import annotations

from typing import Any, Tuple

import numpy as np

from .base_layer import BaseLayer, LayerInput


class BatchNorm(BaseLayer):
    reset = ("std", "norm", "scaled_norm")

    __slots__ = ("gamma", "beta", "norm", "scaled_norm", "mean_mva", "std_mva")

    def __init__(
        self,
        ip: LayerInput,
        axis: int = 0,
        momentum: float = 0.5,
        epsilon: float = 1e-7,
        name: str = None,
    ) -> None:
        self.axis = axis
        self.momentum = momentum
        self.epsilon = epsilon

        self.gamma = self.beta = None

        params = ["gamma", "beta"]

        super().__init__(ip=ip, params=params, name=name)

        ip_shape = self.input_shape()
        self.axes = tuple(ax for ax, _ in enumerate(ip_shape) if ax != axis)

        self.std = None
        self.norm = None
        self.scaled_norm = None

        self.mean_mva = None
        self.std_mva = None

    def fans(self) -> Tuple[int, int]:
        _, ip_fan_out = self.ip_layer.fans()

        return ip_fan_out, ip_fan_out

    def x_dim(self):
        return self.input_shape()[self.axis]

    def build(self) -> Any:
        x_dim = self.x_dim()
        rem_dims = (1,) * len(self.axes)

        shape = (x_dim, *rem_dims)

        self.gamma = self._add_param(shape=shape, initializer="ones")

        self.beta = self._add_param(shape=shape, initializer="zeros")

        self.mean_mva = self._add_param(shape=shape, initializer="zeros")

        self.std_mva = self._add_param(shape=shape, initializer="ones")

    def count_params(self) -> int:
        return 2 * self.x_dim()

    def output(self) -> np.ndarray:
        return self.scaled_norm

    def output_shape(self) -> Tuple:
        return self.input_shape()

    def _update_mva(self, mean: np.ndarray, std: np.ndarray) -> None:
        mom, one_minus_mom = self.momentum, 1 - self.momentum

        self.mean_mva *= mom
        self.mean_mva += one_minus_mom * mean

        self.std_mva *= mom
        self.std_mva += one_minus_mom * std

    def forward_step(self, *args, **kwargs) -> np.ndarray:
        ip = self.input()

        if self.is_training:
            mean = ip.mean(axis=self.axes, keepdims=True)
            std = np.sqrt(ip.var(axis=self.axes, keepdims=True) + self.epsilon)

            self._update_mva(mean, std)

            self.std = std
        else:
            mean, std = self.mean_mva, self.std_mva

        self.norm = ip - mean
        self.norm /= std

        self.scaled_norm = self.gamma * self.norm + self.beta

        return self.scaled_norm

    def backprop_parameters(self, grad: np.ndarray, *args, **kwargs) -> None:
        self.gradients = {
            "gamma": np.sum(grad * self.norm, axis=self.axes, keepdims=True),
            "beta": np.sum(grad, axis=self.axes, keepdims=True),
        }

    def backprop_inputs(self, grad, *args, **kwargs) -> np.ndarray:
        grad *= self.gamma

        mean_share = grad.sum(axis=self.axes, keepdims=True)
        var_share = self.norm * np.sum(grad * self.norm, axis=self.axes, keepdims=True)

        scale = grad.size / self.x_dim()

        return (scale * grad - mean_share - var_share) / (scale * self.std)
