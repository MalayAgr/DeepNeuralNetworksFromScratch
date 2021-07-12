from functools import cached_property

import numpy as np
from dnn.layers.base_layer import BaseLayer
from dnn.layers.utils import (
    accumulate_dX_conv,
    add_activation,
    compute_conv_output_dim,
    compute_conv_padding,
    pad,
    vectorize_for_conv,
)


class Conv2D(BaseLayer):
    reset = (
        "convolutions",
        "activations",
        "_vectorized_ip",
        "_vectorized_kernel",
    )

    str_attrs = ("filters", "kernel_size", "stride", "padding", "activation")

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
