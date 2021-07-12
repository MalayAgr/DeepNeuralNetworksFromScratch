from functools import cached_property
from dnn.layers.base_layer import BaseLayer
from dnn.layers.utils import add_activation
import numpy as np


class Dense(BaseLayer):
    reset = ("linear", "activations")

    def __init__(self, ip, units, activation=None, initializer="he", use_bias=True):
        self.units = units
        self.activation = add_activation(activation)
        self.initializer = initializer

        params = ["weights"]

        self.use_bias = use_bias

        if use_bias is True:
            params.append("biases")

        self.linear = None
        self.activations = None

        super().__init__(ip=ip, params=params)

    @cached_property
    def fans(self):
        fan_in = self.input_shape()[0]
        return fan_in, self.units

    def init_params(self):
        y_dim, _ = self.fans

        variance = self._initializer_variance(self.initializer)

        self.weights = np.random.randn(self.units, y_dim) * np.sqrt(variance)
        self.weights = self.weights.astype(np.float32)

        if self.use_bias:
            self.biases = np.zeros(shape=(self.units, 1), dtype=np.float32)

    def count_params(self):
        total = self.weights.shape[0] * self.weights.shape[-1]

        if self.use_bias:
            return total + self.units

        return total

    def build(self):
        self.init_params()

    def output(self):
        return self.activations

    def output_shape(self):
        if self.activations is not None:
            return self.activations.shape

        return self.units, None

    def forward_step(self, *args, **kwargs):
        linear = np.matmul(self.weights, self.input(), dtype=np.float32)

        if self.use_bias:
            linear += self.biases

        activations = (
            self.activation.forward_step(ip=linear)
            if self.activation is not None
            else linear
        )

        self.linear, self.activations = linear, activations

        return self.activations

    def backprop_step(self, dA, *args, **kwargs):
        ip = self.input()
        m = ip.shape[-1]

        dZ = (
            self.activation.backprop_step(dA, ip=self.linear)
            if self.activation is not None
            else dA
        )

        reg_param = kwargs.pop("reg_param", 0.0)

        dW = np.matmul(dZ, ip.T, dtype=np.float32) / m

        if reg_param > 0:
            dW += (reg_param / m) * self.weights

        self.gradients["weights"] = dW

        if self.use_bias:
            self.gradients["biases"] = np.sum(dZ, keepdims=True, axis=1) / m

        if self.requires_dX is False:
            self.reset_attrs()
            return

        dX = np.matmul(self.weights.T, dZ, dtype=np.float32)

        self.reset_attrs()

        return dX
