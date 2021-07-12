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


class BatchNorm(BaseLayer):
    reset = ("std", "norm", "scaled_norm")

    def __init__(self, ip, axis=0, momentum=0.5, epsilon=1e-7):
        self.axis = axis
        self.momentum = momentum
        self.epsilon = epsilon

        params = ["gamma", "beta"]

        super().__init__(ip=ip, params=params)

        ip_shape = self.input_shape()
        self.x_dim = ip_shape[axis]
        self.axes = tuple(ax for ax, _ in enumerate(ip_shape) if ax != axis)

        self.std = None
        self.norm = None
        self.scaled_norm = None

        self.mean_mva = None
        self.std_mva = None

    @cached_property
    def fans(self):
        _, ip_fan_out = self.ip_layer.fans

        return ip_fan_out, ip_fan_out

    def init_params(self):
        rem_dims = (1,) * len(self.axes)

        self.gamma = np.ones(shape=(self.x_dim, *rem_dims), dtype=np.float32)

        self.beta = np.zeros(shape=(self.x_dim, *rem_dims), dtype=np.float32)

    def _init_mva(self):
        rem_dims = (1,) * len(self.axes)

        self.mean_mva = np.zeros(shape=(self.x_dim, *rem_dims), dtype=np.float32)

        self.std_mva = np.ones(shape=(self.x_dim, *rem_dims), dtype=np.float32)

    def count_params(self):
        return 2 * self.x_dim

    def build(self):
        self.init_params()
        self._init_mva()

    def output(self):
        return self.scaled_norm

    def output_shape(self):
        return self.input_shape()

    def _update_mva(self, mean, std):
        mom, one_minus_mom = self.momentum, 1 - self.momentum

        self.mean_mva *= mom
        self.mean_mva += one_minus_mom * mean

        self.std_mva *= mom
        self.std_mva += one_minus_mom * mean

    def forward_step(self, *args, **kwargs):
        ip = self.input()

        if self.is_training:
            mean = ip.mean(axis=self.axes, keepdims=True)
            std = np.sqrt(ip.var(axis=self.axes, keepdims=True) + self.epsilon)

            self._update_mva(mean, std)

            self.std = std
        else:
            mean, std = self.mean_mva, self.std_mva

        self.norm = ip - mean
        self.norm /= std

        self.scaled_norm = self.gamma * self.norm + self.beta

        return self.scaled_norm

    def backprop_step(self, dA, *args, **kwargs):
        self.gradients = {
            "gamma": np.sum(dA * self.norm, axis=self.axes, keepdims=True),
            "beta": np.sum(dA, axis=self.axes, keepdims=True),
        }

        dA *= self.gamma

        mean_share = dA.sum(axis=self.axes, keepdims=True)
        var_share = self.norm * np.sum(dA * self.norm, axis=self.axes, keepdims=True)

        scale = dA.size / self.x_dim

        if self.requires_dX is False:
            self.reset_attrs()
            return

        dX = (scale * dA - mean_share - var_share) / (scale * self.std)

        self.reset_attrs()

        return dX


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


class Conv2D(BaseLayer):
    reset = (
        "convolutions",
        "activations",
        "_vectorized_ip",
        "_vectorized_kernel",
    )

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
        self.kernels = self.kernels.astype(np.float32)

        if self.use_bias:
            self.biases = np.zeros(shape=(self.filters, 1, 1, 1), dtype=np.float32)

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

        X = vectorize_for_conv(
            X, self.kernel_size, self.stride, (self.out_H, self.out_W)
        )

        self._vectorized_ip = np.moveaxis(X, -1, 0)

        weights = self._vectorize_kernels()

        convolution = np.matmul(
            self._vectorized_ip, weights[None, ...], dtype=np.float32
        )

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

        dW = np.matmul(ip[..., None], dZ[..., None, :], dtype=np.float32).sum(
            axis=(0, 1)
        )

        return dW.reshape(-1, self.kernel_H, self.kernel_W, self.filters) / ip.shape[0]

    def backprop_step(self, dA, *args, **kwargs):
        dZ = (
            self.activation.backprop_step(dA, ip=self.convolutions)
            if self.activation is not None
            else dA
        )

        dZ = np.swapaxes(dZ, 0, -1).reshape(dZ.shape[-1], -1, self.filters)

        dW = self._compute_dW(dZ)

        reg_param = kwargs.pop("reg_param", 0.0)
        if reg_param > 0:
            dW += (reg_param / dZ.shape[-1]) * self.kernels

        self.gradients["kernels"] = dW

        if self.use_bias:
            self.gradients["biases"] = (
                dZ.sum(axis=(0, 1)).reshape(-1, *self.biases.shape[1:]) / dZ.shape[-1]
            )

        if self.requires_dX is False:
            self.reset_attrs()
            return

        dX = accumulate_dX_conv(
            dX_shape=(dZ.shape[0], self.ip_C, *self._padded_shape),
            output_size=(self.out_H, self.out_W),
            dIp=np.matmul(dZ, self._vectorized_kernel.T, dtype=np.float32),
            stride=self.stride,
            kernel_size=self.kernel_size,
            reshape=(-1, self.ip_C, self.kernel_H, self.kernel_W),
            padding=(self.p_H, self.p_W),
        )

        self.reset_attrs()

        return dX


class MaxPooling2D(BaseLayer):
    reset = ("pooled", "_dX_share")

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

        X = vectorize_for_conv(
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

    def backprop_step(self, dA, *args, **kwargs):
        dA = np.swapaxes(dA, 0, -1).reshape(dA.shape[-1], -1, self.windows)

        if self.requires_dX is False:
            self.reset_attrs()
            return

        dX = accumulate_dX_conv(
            dX_shape=(dA.shape[0], self.windows, *self._padded_shape),
            output_size=(self.out_H, self.out_W),
            dIp=self._dX_share * dA[..., None],
            stride=self.stride,
            kernel_size=self.pool_size,
            reshape=(-1, self.windows, self.pool_H, self.pool_W),
            padding=(self.p_H, self.p_W),
        )

        self.reset_attrs()

        return dX


class AveragePooling2D(MaxPooling2D):
    def _get_pool_outputs(self, ip):
        ip_shape = ip.shape

        averages = ip.mean(axis=-1)

        distributed = (
            np.ones(shape=(1, 1, 1, ip_shape[-1]), dtype=np.float32) / ip_shape[-1]
        )

        shape = (self.windows, self.out_H, self.out_W, -1)

        return np.swapaxes(averages, 0, -1).reshape(*shape), distributed


class Flatten(BaseLayer):
    reset = ("flat",)

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

        return self._units, None

    def forward_step(self, *args, **kwargs):
        self.flat = self.input().reshape(self._units, -1)

        return self.flat

    def backprop_step(self, dA, *args, **kwargs):
        dA = dA.reshape(*self._ip_dims, -1)

        self.reset_attrs()

        return dA


class Dropout(BaseLayer):
    reset = ("dropped", "dropout_mask")

    def __init__(self, ip, keep_prob=0.5):
        if not 0 < keep_prob <= 1:
            raise AttributeError("keep_prob should be in the interval (0, 1]")

        self.keep_prob = keep_prob

        super().__init__(ip=ip, trainable=False)

        self.dropped = None
        self.dropout_mask = None

    @cached_property
    def fans(self):
        _, ip_fan_out = self.ip_layer.fans

        return ip_fan_out, ip_fan_out

    def output(self):
        return self.dropped

    def output_shape(self):
        return self.input_shape()

    def forward_step(self, *args, **kwargs):
        ip = self.input()

        self.dropout_mask = (
            np.random.rand(*ip.shape).astype(np.float32) < self.keep_prob
        )

        self.dropped = (ip * self.dropout_mask) / self.keep_prob

        return self.dropped

    def backprop_step(self, dA, *args, **kwargs):
        dA *= self.dropout_mask
        dA /= self.keep_prob

        self.reset_attrs()

        return dA
