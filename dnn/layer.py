from functools import cached_property

import numpy as np

from dnn.base_layer import BaseLayer
from dnn.utils import (
    add_activation,
    compute_conv_output_dim,
    compute_conv_padding,
    pad,
    vectorize_for_conv,
    accumulate_dX_conv,
)


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
        self.p_H, self.p_W = compute_conv_padding(kernel_size, mode=padding)

        self.initializer = initializer

        self.activation = add_activation(activation)

        params = ["kernels"]

        self.use_bias = use_bias
        if use_bias:
            params.append("biases")

        super().__init__(ip=ip, params=params)

        self.ip_C, self.ip_H, self.ip_W = self.input_shape()[:-1]

        self.out_H = compute_conv_output_dim(
            self.ip_H, self.kernel_H, self.p_H, self.stride_H
        )
        self.out_W = compute_conv_output_dim(
            self.ip_W, self.kernel_W, self.p_W, self.stride_W
        )

        self.convolutions = None
        self.activations = None

        self._vectorized_ip = None
        self._vectorized_kernel = None
        self._slice_idx = None
        self._padded_shape = None

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

    def _vectorize_kernels(self):
        self._vectorized_kernel = self.kernels.reshape(-1, self.filters)
        return self._vectorized_kernel

    def _convolve(self, X):
        X, self._padded_shape = pad(X, self.p_H, self.p_W)

        X, self._slice_idx = vectorize_for_conv(
            X, self.kernel_size, self.stride, (self.out_H, self.out_W)
        )

        self._vectorized_ip = np.moveaxis(X, -1, 0)

        weights = self._vectorize_kernels()

        convolution = np.matmul(self._vectorized_ip, weights[None, ...])

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
        dVec_ip = np.matmul(dZ, self._vectorized_kernel.T)

        dX = np.zeros(
            shape=(dZ.shape[0], self.ip_C, *self._padded_shape), dtype=np.float32
        )

        shape = (-1, self.ip_C, self.kernel_H, self.kernel_W)

        return accumulate_dX_conv(
            dX=dX,
            dIp=dVec_ip,
            slice_idx=self._slice_idx,
            kernel_size=self.kernel_size,
            shape=shape,
            padding=(self.p_H, self.p_W),
        )

    def backprop_step(self, dA, *args, **kwargs):
        reg_param = kwargs.pop("reg_param", 0.0)

        dZ = (
            self.activation.backprop_step(dA, ip=self.convolutions)
            if self.activation is not None
            else dA
        )

        dZ_flat = np.swapaxes(dZ, 0, -1).reshape(dZ.shape[-1], -1, self.filters)

        dW = self._compute_dW(dZ_flat)

        self.gradients["kernels"] = (
            dW + (reg_param / dZ.shape[-1]) * self.kernels if reg_param > 0 else dW
        )

        if self.use_bias:
            self.gradients["biases"] = dZ_flat.sum(axis=(0, 1)) / dZ.shape[-1]

        self.dX = self._compute_dX(dZ_flat)

        return self.dX


class MaxPooling2D(BaseLayer):
    def __init__(self, ip, pool_size, stride=(2, 2), padding="valid"):
        self.pool_size = pool_size
        self.pool_H, self.pool_W = pool_size

        self.stride = stride
        self.stride_H, self.stride_W = stride

        self.padding = padding
        self.p_H, self.p_W = compute_conv_padding(pool_size, mode=padding)

        super().__init__(ip, trainable=False)

        self.windows, self.ip_H, self.ip_W = self.input_shape()[:-1]

        self.out_H = compute_conv_output_dim(
            self.ip_H, self.pool_H, self.p_H, self.stride_H
        )
        self.out_W = compute_conv_output_dim(
            self.ip_W, self.pool_W, self.p_W, self.stride_W
        )

        self.pooled = None

        self._slice_idx = None
        self._padded_shape = None
        self._dX_share = None

    @cached_property
    def fans(self):
        return self.ip_C, self.ip_C

    def output(self):
        return self.pooled

    def output_shape(self):
        if self.pooled is not None:
            return self.pooled.shape

        return self.windows, self.out_H, self.out_W, None

    def _get_pool_outputs(self, ip):
        ip_shape = ip.shape

        flat = np.prod(ip_shape[:-1])
        p_area = ip_shape[-1]

        ip_idx = np.arange(flat)

        max_idx = ip.argmax(axis=-1).ravel()

        maximums = ip.reshape(-1, p_area)[ip_idx, max_idx]

        mask = np.zeros(shape=(flat, p_area), dtype=bool)
        mask[ip_idx, max_idx] = True

        maximums = maximums.reshape(*ip_shape[:-1])
        mask = mask.reshape(*ip_shape)

        shape = (self.windows, self.out_H, self.out_W, -1)

        return np.swapaxes(maximums, 0, -1).reshape(*shape), mask

    def _pool(self, X):
        X, self._padded_shape = pad(X, self.p_H, self.p_W)

        X, self._slice_idx = vectorize_for_conv(
            X=X,
            kernel_size=self.pool_size,
            stride=self.stride,
            output_size=(self.out_H, self.out_W),
            reshape=(self.windows, self.pool_H * self.pool_W, X.shape[-1]),
        )

        X = np.moveaxis(X, -1, 0)

        pooled, self._dX_share = self._get_pool_outputs(ip=X)

        return pooled

    def forward_step(self, *args, **kwargs):
        self.pooled = self._pool(self.input())
        return self.pooled

    def _compute_dX(self, dZ):
        dVec_ip = self._dX_share * dZ[..., None]

        dX = np.zeros(
            shape=(dZ.shape[0], self.windows, *self._padded_shape), dtype=np.float32
        )

        shape = (-1, self.windows, self.pool_H, self.pool_W)

        return accumulate_dX_conv(
            dX=dX,
            dIp=dVec_ip,
            slice_idx=self._slice_idx,
            kernel_size=self.pool_size,
            shape=shape,
            padding=(self.p_H, self.p_W),
        )

    def backprop_step(self, dA, *args, **kwargs):
        dA_flat = np.swapaxes(dA, 0, -1).reshape(dA.shape[-1], -1, self.windows)
        self.dX = self._compute_dX(dA_flat)
        return self.dX


class AveragePooling2D(MaxPooling2D):
    def _get_pool_outputs(self, ip):
        ip_shape = ip.shape

        averages = ip.mean(axis=-1)

        distributed = np.ones_like(ip) / ip_shape[-1]

        shape = (self.windows, self.out_H, self.out_W, -1)

        return np.swapaxes(averages, 0, -1).reshape(*shape), distributed


class Flatten(BaseLayer):
    def __init__(self, ip):
        super().__init__(ip=ip, trainable=False)

        self._ip_dims = self.input_shape()[:-1]

        self._units = np.prod(self._ip_dims)

        self.flat = None

    @cached_property
    def fans(self):
        _, ip_fan_out = self.ip_layer.fans

        return ip_fan_out, self.units

    def output(self):
        return self.flat

    def output_shape(self):
        if self.flat is not None:
            return self.flat.shape

        return self.units, None

    def forward_step(self, *args, **kwargs):
        self.flat = self.input().reshape(self._units, -1)

        return self.flat

    def backprop_step(self, dA, *args, **kwargs):
        self.dX = dA.reshape(*self._ip_dims, -1)

        return self.dX
