from functools import cached_property

import numpy as np

from dnn.activations import Activation
from dnn.base_layer import BaseLayer
from dnn.input_layer import Input
from dnn.utils import activation_factory


class BatchNorm:
    def __init__(self, ip, epsilon=1e-7, momentum=0.5):
        if not isinstance(ip, Layer):
            raise AttributeError("ip should be an instance of Layer")

        self.ip_layer = ip
        self.epsilon = epsilon

        self.gamma, self.beta = self.init_params()

        self.std = None
        self.Z_hat = None
        self.norm = None

        self.mean_ewa, self.std_ewa = self.init_ewa()
        self.momentum = momentum

    def init_params(self):
        gamma = np.ones(shape=(self.ip_layer.units, 1))
        beta = np.zeros(shape=(self.ip_layer.units, 1))
        return gamma, beta

    def init_ewa(self):
        mean = np.zeros(shape=(self.ip_layer.units, 1), dtype=np.float64)
        std = np.ones(shape=(self.ip_layer.units, 1), dtype=np.float64)

        return mean, std

    def update_ewa(self, mean, std):
        self.mean_ewa = self.momentum * self.mean_ewa + (1 - self.momentum) * mean
        self.std_ewa = self.momentum * self.std_ewa + (1 - self.momentum) * std

    def forward_step(self, X):
        if self.ip_layer.is_training:
            mean = np.mean(X, axis=1, keepdims=True)

            var = np.var(X, axis=1, keepdims=True) + self.epsilon
            std = np.sqrt(var)

            self.update_ewa(mean, std)

            self.std = std
        else:
            mean = self.mean_ewa
            std = self.std_ewa

        Z_hat = np.divide(X - mean, std)
        self.Z_hat = Z_hat

        self.norm = self.gamma * Z_hat + self.beta
        return self.norm

    def backprop_step(self, dA, bs):
        activation_grads = self.ip_layer.activation.calculate_derivatives(self.norm)

        d_norm = self.ip_layer.compute_dZ(dA, activation_grads)

        grads = {
            "gamma": np.sum(d_norm * self.Z_hat, axis=1, keepdims=True),
            "beta": np.sum(d_norm, axis=1, keepdims=True),
        }

        self.ip_layer.gradients.update(grads)

        dZ_hat = d_norm * self.gamma

        dZ_hat_sum = np.sum(dZ_hat, axis=1, keepdims=True)
        dZ_hat_prod = self.Z_hat * np.sum(dZ_hat * self.Z_hat, axis=1, keepdims=True)

        return (bs * dZ_hat - dZ_hat_sum - dZ_hat_prod) / (bs * self.std)


class Dense(BaseLayer):
    def __init__(self, ip, units, activation=None, initializer="he", use_bias=True):
        self.units = units
        self.activation = self._add_activation(activation)
        self.initializer = initializer

        params = ["weights"]

        self.use_bias = use_bias

        if use_bias is True:
            params.append("biases")

        super().__init__(ip=ip, params=params, linear=None, activations=None)

    @staticmethod
    def _add_activation(activation):
        if activation is None:
            return

        if isinstance(activation, Activation):
            return activation

        return activation_factory(activation)

    @cached_property
    def fans(self):
        fan_in = self.input_shape()[0]
        return fan_in, self.units

    def init_params(self):
        y_dim, _ = self.fans

        variance = self._initializer_variance(self.initializer)

        self.weights = np.random.randn(self.units, y_dim) * np.sqrt(variance)

        if self.use_bias:
            self.biases = np.zeros(shape=(self.units, 1))

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
        linear = np.matmul(self.weights, self.input())

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

        gradients = {}

        gradients["weights"] = (
            (np.matmul(dZ, ip.T) + reg_param * self.weights) / m
            if reg_param > 0
            else np.matmul(dZ, ip.T) / m
        )

        if self.use_bias:
            gradients["biases"] = np.sum(dZ, keepdims=True, axis=1) / m

        self.gradients.update(gradients)

        self.dX = np.matmul(self.weights.T, dZ)

        return self.dX
