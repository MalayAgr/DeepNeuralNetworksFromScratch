from typing import Any, Optional, Tuple, Union

import numpy as np

from .activations import Activation
from .base_layer import BaseLayer, LayerInput
from .utils import (
    accumulate_dX_conv,
    add_activation,
    compute_conv_output_dim,
    compute_conv_padding,
    convolve2d,
)


class Conv2D(BaseLayer):
    reset = (
        "convolutions",
        "activations",
        "_vec_ip",
        "_vec_kernel",
    )

    str_attrs = ("filters", "kernel_size", "stride", "padding", "activation")

    def __init__(
        self,
        ip: LayerInput,
        filters: int,
        kernel_size: Tuple[int, int],
        stride: Tuple[int, int] = (1, 1),
        activation: Optional[Union[Activation, str]] = None,
        padding: str = "valid",
        initializer: str = "he",
        use_bias: bool = True,
    ) -> None:
        self.filters = filters

        self.kernel_size = kernel_size
        self.kernel_H, self.kernel_W = kernel_size

        self.stride = stride
        self.stride_H, self.stride_W = stride

        self.padding = padding
        self.p_H, self.p_W = compute_conv_padding(kernel_size, mode=padding)

        self.initializer = initializer

        self.activation = add_activation(activation)

        self.kernels = None
        params = ["kernels"]

        self.use_bias = use_bias
        if use_bias:
            self.biases = None
            params.append("biases")

        super().__init__(ip=ip, params=params)

        self.ip_C = self.input_shape()[0]

        self.convolutions = None
        self.activations = None

        self._vec_ip = None
        self._vec_kernel = None

    def fans(self) -> Tuple[int, int]:
        receptive_field_size = np.prod(self.kernel_size)
        fan_in = self.ip_C * receptive_field_size
        return fan_in, receptive_field_size * self.filters

    def build(self) -> Any:
        shape = (self.ip_C, *self.kernel_size, self.filters)

        self.kernels = self._add_param(shape=shape, initializer=self.initializer)

        if self.use_bias:
            shape = (self.filters, 1, 1, 1)
            self.biases = self._add_param(shape=shape, initializer="zeros")

    def count_params(self) -> int:
        total = np.prod(self.kernels.shape)

        if self.use_bias:
            return total + self.biases.shape[0]

        return total

    def output(self) -> np.ndarray:
        return self.activations

    def output_area(self) -> Tuple[int, int]:
        ip_shape = self.input_shape()
        ipH, ipW = ip_shape[1], ip_shape[2]

        oH = compute_conv_output_dim(ipH, self.kernel_H, self.p_H, self.stride_H)
        oW = compute_conv_output_dim(ipW, self.kernel_W, self.p_W, self.stride_W)

        return oH, oW

    def output_shape(self) -> Tuple:
        if self.activations is not None:
            return self.activations.shape

        oH, oW = self.output_area()

        return self.filters, oH, oW, None

    def forward_step(self, *args, **kwargs) -> np.ndarray:
        self.convolutions, self._vec_ip, self._vec_kernel = convolve2d(
            X=self.input(),
            kernel=self.kernels,
            stride=self.stride,
            padding=(self.p_H, self.p_W),
            return_vec_ip=True,
            return_vec_kernel=True,
        )

        if self.use_bias:
            self.convolutions += self.biases

        self.activations = self.activation.forward_step(ip=self.convolutions)

        return self.activations

    def _param_scale(self) -> int:
        oH, oW = self.output_area()
        return self._vec_ip.shape[0] * oH * oW

    def _compute_dW(self, dZ: np.ndarray) -> np.ndarray:
        ip = self._vec_ip

        dW = np.matmul(ip[..., None], dZ[..., None, :], dtype=np.float32).sum(
            axis=(0, 1)
        )

        scale = self._param_scale()

        return dW.reshape(-1, self.kernel_H, self.kernel_W, self.filters) / scale

    def backprop_step(self, dA: np.ndarray, *args, **kwargs) -> np.ndarray:
        dA = self.activation.backprop_step(dA=dA, ip=self.convolutions)

        dA = np.swapaxes(dA, 0, -1).reshape(dA.shape[-1], -1, self.filters)

        dW = self._compute_dW(dA)

        scale = self._param_scale()

        reg_param = kwargs.pop("reg_param", 0.0)
        if reg_param > 0:
            dW += (reg_param / scale) * self.kernels

        self.gradients["kernels"] = dW

        if self.use_bias:
            self.gradients["biases"] = (
                dA.sum(axis=(0, 1)).reshape(-1, *self.biases.shape[1:]) / scale
            )

        if self.requires_dX is False:
            self.reset_attrs()
            return

        ipH, ipW = self.input_shape()[1:-1]
        padded_shape = (ipH + 2 * self.p_H, ipW + 2 * self.p_W)

        dX = accumulate_dX_conv(
            dX_shape=(dA.shape[0], self.ip_C, *padded_shape),
            output_size=self.output_area(),
            dIp=np.matmul(dA, self._vec_kernel.T, dtype=np.float32),
            stride=self.stride,
            kernel_size=self.kernel_size,
            reshape=(-1, self.ip_C, self.kernel_H, self.kernel_W),
            padding=(self.p_H, self.p_W),
        )

        self.reset_attrs()

        return dX
