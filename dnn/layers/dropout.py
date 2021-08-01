from __future__ import annotations
from typing import Optional, Tuple

import numpy as np

from .base_layer import BaseLayer, LayerInput


class Dropout(BaseLayer):
    reset = ("dropped", "dropout_mask")
    str_attrs = ("keep_prob",)

    def __init__(
        self, ip: LayerInput, keep_prob: float = 0.5, name: str = None
    ) -> None:
        if not 0 < keep_prob <= 1:
            raise AttributeError("keep_prob should be in the interval (0, 1]")

        self.keep_prob = keep_prob

        super().__init__(ip=ip, trainable=False, name=name)

        self.dropped = None
        self.dropout_mask = None

    def fans(self) -> Tuple[int, int]:
        _, ip_fan_out = self.ip_layer.fans()

        return ip_fan_out, ip_fan_out

    def output(self) -> Optional[np.ndarray]:
        return self.dropped

    def output_shape(self) -> Tuple:
        return self.input_shape()

    def forward_step(self, *args, **kwargs) -> np.ndarray:
        ip = self.input()

        self.dropout_mask = (
            np.random.rand(*ip.shape).astype(np.float32) < self.keep_prob
        )

        self.dropped = (ip * self.dropout_mask) / self.keep_prob

        return self.dropped

    def backprop_inputs(self, grad: np.ndarray, *args, **kwargs) -> np.ndarray:
        grad *= self.dropout_mask
        grad /= self.keep_prob

        return grad
