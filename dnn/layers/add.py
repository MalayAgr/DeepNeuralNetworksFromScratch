from typing import List, Tuple

import numpy as np
from .base_layer import BaseLayer, LayerInput, MultiInputBaseLayer


class Add(MultiInputBaseLayer):
    def __init__(self, ip: List[LayerInput], name: str = None) -> None:
        super().__init__(ip=ip, trainable=False, name=name)

        self.add = None

    def output(self) -> np.ndarray:
        return self.add

    def output_shape(self) -> Tuple:
        return self.input_shape()

    def forward_step(self, *args, **kwargs) -> np.ndarray:
        self.add = np.add.reduce(self.input())
        return self.add

    def backprop_inputs(self, grad: np.ndarray, *args, **kwargs) -> Tuple[np.ndarray]:
        num_ips = len(self.input())
        return tuple(grad.copy() for _ in range(num_ips))
