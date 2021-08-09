from typing import List, Optional, Tuple

import numpy as np
from .base_layer import LayerInput, MultiInputBaseLayer


class Add(MultiInputBaseLayer):
    def __init__(self, ip: List[LayerInput], name: str = None) -> None:
        super().__init__(ip=ip, trainable=False, name=name)

        self.added: np.ndarray

    def _validate_same_shape(self) -> None:
        shapes = self.input_shape()
        first_shape = shapes[0]
        if any(shape != first_shape for shape in shapes[1:]):
            msg = (
                "All inputs must output the same shape. "
                f"Expected shape: {first_shape}"
            )
            raise ValueError(msg)

    def output(self) -> Optional[np.ndarray]:
        return self.added

    def output_shape(self) -> Tuple:
        return self.input_shape()[0]

    def forward_step(self, *args, **kwargs) -> np.ndarray:
        self._validate_same_shape()
        self.added = np.add.reduce(self.input())
        return self.add

    def backprop_inputs(self, grad: np.ndarray, *args, **kwargs) -> Tuple[np.ndarray]:
        num_ips = len(self.input())
        return tuple(grad.copy() for _ in range(num_ips))
