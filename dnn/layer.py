from functools import cached_property

import numpy as np

from dnn.base_layer import BaseLayer
from dnn.utils import add_activation


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

        self.gradients["weights"] = (
            (np.matmul(dZ, ip.T) + reg_param * self.weights) / m
            if reg_param > 0
            else np.matmul(dZ, ip.T) / m
        )

        if self.use_bias:
            self.gradients["biases"] = np.sum(dZ, keepdims=True, axis=1) / m

        self.dX = np.matmul(self.weights.T, dZ)

        return self.dX


class Conv2D(BaseLayer):
    def __init__(
        self,
        ip,
        filters,
        kernel_size,
        stride=(1, 1),
        activation=None,
        padding="valid",
        initializer="he",
        use_bias=True,
    ):
        self.filters = filters

        self.kernel_size = kernel_size
        self.kernel_H, self.kernel_W = kernel_size

        self.stride = stride
        self.stride_H, self.stride_W = stride

        self.padding = padding
        self.p_H, self.p_W = self._get_padding(*kernel_size)

        self.initializer = initializer

        self.activation = add_activation(activation)

        params = ["kernels"]

        self.use_bias = use_bias
        if use_bias:
            params.append("biases")

        super().__init__(ip=ip, params=params)

        self.ip_C, self.ip_H, self.ip_W = self.input_shape()[:-1]

        self.out_H = self._get_output_dim(
            self.ip_H, self.kernel_H, self.p_H, self.stride_H
        )
        self.out_W = self._get_output_dim(
            self.ip_W, self.kernel_W, self.p_W, self.stride_W
        )

        self.convolutions = None
        self.activations = None

        self._vectorized_ip = None
        self._vectorized_kernel = None
        self._slice_idx = None
        self._padded_shape = None

    def _get_padding(self, kH, kW):
        if self.padding == "same":
            p_H = int(np.ceil((kH - 1) / 2))
            p_W = int(np.ceil((kW - 1) / 2))

            return p_H, p_W

        return 0, 0

    def _get_output_dim(self, n, f, p, s):
        return int((n - f + 2 * p) / s + 1)

    @cached_property
    def fans(self):
        receptive_field_size = np.prod(self.kernel_size)
        fan_in = self.ip_C * receptive_field_size
        return fan_in, receptive_field_size * self.filters

    def init_params(self):
        variance = self._initializer_variance(self.initializer)

        shape = (self.ip_C, *self.kernel_size, self.filters)

        self.kernels = np.random.randn(*shape) * np.sqrt(variance)

        if self.use_bias:
            self.biases = np.zeros(shape=(self.filters, 1, 1, 1))

    def count_params(self):
        total = np.prod(self.kernels.shape)

        if self.use_bias:
            return total + self.filters

        return total

    def build(self):
        self.init_params()

    def output(self):
        return self.activations

    def output_shape(self):
        if self.activations is not None:
            return self.activations.shape

        return self.filters, self.out_H, self.out_W, None

    def _pad(self, X):
        padded = np.pad(X, ((0, 0), (self.p_H, self.p_H), (self.p_W, self.p_W), (0, 0)))

        self._padded_shape = padded.shape[1], padded.shape[2]

        return padded

    def _vectorize_ip(self, X):
        self._slice_idx = np.array(
            [
                (i * self.stride_H, j * self.stride_W)
                for i in range(self.out_H)
                for j in range(self.out_W)
            ]
        )

        shape = (-1, X.shape[-1])

        self._vectorized_ip = np.array(
            [
                X[:, i : i + self.kernel_H, j : j + self.kernel_W].reshape(shape)
                for i, j in self._slice_idx
            ]
        ).transpose(2, 0, 1)

        return self._vectorized_ip

    def _vectorize_kernels(self):
        self._vectorized_kernel = self.kernels.reshape(-1, self.filters)
        return self._vectorized_kernel

    def _convolve(self, X):
        X = self._pad(X)
        X = self._vectorize_ip(X)

        weights = self._vectorize_kernels()

        convolution = np.matmul(X, weights[None, ...])

        shape = (self.filters, self.out_H, self.out_W, -1)

        return np.swapaxes(convolution, 0, 2).reshape(shape)

    def forward_step(self, *args, **kwargs):
        convolutions = self._convolve(self.input())

        if self.use_bias:
            convolutions += self.biases

        activations = (
            self.activation.forward_step(ip=convolutions)
            if self.activation is not None
            else convolutions
        )

        self.convolutions, self.activations = convolutions, activations

        return self.activations

    def _compute_dW(self, dZ):
        ip = self._vectorized_ip

        dW = np.matmul(ip[..., None], dZ[..., None, :]).sum(axis=(0, 1))

        return dW.reshape(-1, self.kernel_H, self.kernel_W, self.filters) / ip.shape[0]

    def _compute_dX(self, dZ):
        kernels = self._vectorized_kernel

        dVec_ip = np.matmul(dZ, kernels.T)

        dX = np.zeros(shape=(dZ.shape[0], self.ip_C, *self._padded_shape))

        shape = (-1, self.ip_C, self.kernel_H, self.kernel_W)

        for idx, (start_r, start_c) in enumerate(self._slice_idx):
            end_r, end_c = start_r + self.kernel_H, start_c + self.kernel_W
            dX[:, :, start_r:end_r, start_c:end_c] += dVec_ip[:, idx, :].reshape(*shape)

        if self.padding != "valid":
            dX = dX[:, :, self.p_H : -self.p_H, self.p_W : -self.p_W]

        return np.moveaxis(dX, 0, -1)

    def backprop_step(self, dA, *args, **kwargs):
        reg_param = kwargs.pop("reg_param", 0.0)

        dZ = (
            self.activation.backprop_step(dA, ip=self.convolutions)
            if self.activation is not None
            else dA
        )

        dZ_flat = np.swapaxes(dZ, 0, 3).reshape(dZ.shape[-1], -1, self.filters)

        dW = self._compute_dW(dZ_flat)

        self.gradients["kernels"] = (
            dW + (reg_param / dZ.shape[-1]) * self.kernels if reg_param > 0 else dW
        )

        if self.use_bias:
            self.gradients["biases"] = dZ_flat.sum(axis=(0, 1)) / dZ.shape[-1]

        self.dX = self._compute_dX(dZ_flat)

        return self.dX


class Flatten(BaseLayer):
    def __init__(self, ip):
        super().__init__(ip=ip, trainable=False)

        self.ip_dims = self.input_shape()[:-1]

        self.units = np.prod(self.ip_dims)

        self.flat = None

    @cached_property
    def fans(self):
        _, ip_fan_out = self.ip_layer.fans

        return ip_fan_out, self.units

    def count_params(self):
        return 0

    def build(self):
        return

    def output(self):
        return self.flat

    def output_shape(self):
        if self.flat is not None:
            return self.flat.shape

        return self.units, None

    def forward_step(self, *args, **kwargs):
        self.flat = self.input().reshape(self.units, -1)

        return self.flat

    def backprop_step(self, dA, *args, **kwargs):
        self.dX = dA.reshape(*self.ip_dims, -1)

        return self.dX
