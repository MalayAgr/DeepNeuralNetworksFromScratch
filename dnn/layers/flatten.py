from __future__ import annotations
from typing import Tuple

import numpy as np

from .base_layer import BaseLayer, LayerInput


class Flatten(BaseLayer):
    reset = ("flat",)

    def __init__(self, ip: LayerInput) -> None:
        super().__init__(ip=ip, trainable=False)

        self._ip_dims = self.input_shape()[:-1]

        self._units = np.prod(self._ip_dims)

        self.flat = None

    def fans(self) -> Tuple[int, int]:
        _, ip_fan_out = self.ip_layer.fans()

        return ip_fan_out, self.units

    def output(self) -> np.ndarray:
        return self.flat

    def output_shape(self) -> Tuple:
        if self.flat is not None:
            return self.flat.shape

        return self._units, None

    def forward_step(self, *args, **kwargs) -> np.ndarray:
        self.flat = self.input().reshape(self._units, -1)

        return self.flat

    def backprop_step(self, dA: np.ndarray, *args, **kwargs) -> np.ndarray:
        dA = dA.reshape(*self._ip_dims, -1)

        self.reset_attrs()

        return dA
