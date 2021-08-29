from __future__ import annotations
from typing import Optional, Tuple

import numpy as np

from .base_layer import BaseLayer, LayerInput


class Flatten(BaseLayer):
    """Flatten layer.

    Inherits from
    ----------
    BaseLayer

    Attributes
    ----------
    flat: np.ndarray
        Flattened output of the layer.

    Input shape
    ----------
    A n-dimensional Numpy array of shape (..., batch_size).

    Output shape
    ----------
    A two-dimensional Numpy array of shape (x, batch_size),
    where x is the product of all dimensions of the input except batch_size.

    Example
    ----------
    >>> import numpy as np
    >>> from dnn import Input
    >>> from dnn.layers import Flatten

    >>> ip = Input(shape=(3, 6, 6, None)) # Create input
    >>> ip.ip = np.random.rand(3, 6, 6, 64)

    >>> layer = Flatten(ip=ip)
    >>> layer.forward_step().shape # Forward step
    (108, 64)
    """

    reset = ("flat",)

    def __init__(self, ip: LayerInput, name: str = None) -> None:
        super().__init__(ip=ip, trainable=False, name=name)

        self._ip_dims = self.input_shape()[:-1]

        self._units = np.prod(self._ip_dims)

        self.flat = None

    def output(self) -> Optional[np.ndarray]:
        return self.flat

    def output_shape(self) -> Tuple:
        if self.flat is not None:
            return self.flat.shape

        return self._units, None

    def forward_step(self, *args, **kwargs) -> np.ndarray:
        self.flat = self.input().reshape(self._units, -1)

        return self.flat

    def backprop_inputs(self, grad: np.ndarray, *args, **kwargs) -> np.ndarray:
        return grad.reshape(*self._ip_dims, -1)
