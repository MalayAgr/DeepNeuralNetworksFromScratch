from typing import Any, Optional, Tuple, Union

import numpy as np
from dnn.layers.activations import Activation

from .base_layer import BaseLayer, LayerInput
from .utils import add_activation


class Dense(BaseLayer):
    reset = ("linear", "activations")
    str_attrs = ("units", "activation")

    def __init__(
        self,
        ip: LayerInput,
        units: int,
        activation: Optional[Union[Activation, str]] = None,
        initializer: str = "he",
        use_bias: bool = True,
    ) -> None:
        self.units = units
        self.activation = add_activation(activation)
        self.initializer = initializer

        self.weights = None
        params = ["weights"]

        self.use_bias = use_bias

        if use_bias is True:
            self.biases = None
            params.append("biases")

        self.linear = None
        self.activations = None

        super().__init__(ip=ip, params=params)

    def fans(self) -> Tuple[int, int]:
        fan_in = self.input_shape()[0]
        return fan_in, self.units

    def build(self) -> Any:
        y_dim, _ = self.fans()

        shape = (self.units, y_dim)

        self.weights = self._add_param(shape=shape, initializer=self.initializer)

        if self.use_bias:
            shape = (self.units, 1)
            self.biases = self._add_param(shape=shape, initializer="zeros")

    def count_params(self) -> int:
        total = self.weights.shape[0] * self.weights.shape[-1]

        if self.use_bias:
            return total + self.units

        return total

    def output(self) -> np.ndarray:
        return self.activations

    def output_shape(self) -> Tuple:
        if self.activations is not None:
            return self.activations.shape

        return self.units, None

    def forward_step(self, *args, **kwargs) -> np.ndarray:
        self.linear = np.matmul(self.weights, self.input(), dtype=np.float32)

        if self.use_bias:
            self.linear += self.biases

        self.activations = self.activation.forward_step(ip=self.linear)

        return self.activations

    def backprop_step(self, dA: np.ndarray, *args, **kwargs) -> np.ndarray:
        ip = self.input()
        m = ip.shape[-1]

        dA = self.activation.backprop_step(dA, ip=self.linear)

        reg_param = kwargs.pop("reg_param", 0.0)

        dW = np.matmul(dA, ip.T, dtype=np.float32) / m

        if reg_param > 0:
            dW += (reg_param / m) * self.weights

        self.gradients["weights"] = dW

        if self.use_bias:
            self.gradients["biases"] = np.sum(dA, keepdims=True, axis=1) / m

        if self.requires_dX is False:
            self.reset_attrs()
            return

        dX = np.matmul(self.weights.T, dA, dtype=np.float32)

        self.reset_attrs()

        return dX
