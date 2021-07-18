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
    pad,
    vectorize_for_conv,
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

        params = ["kernels"]

        self.use_bias = use_bias
        if use_bias:
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

    def init_params(self) -> None:
        variance = self._initializer_variance(self.initializer)

        shape = (self.ip_C, *self.kernel_size, self.filters)

        self.kernels = np.random.randn(*shape) * np.sqrt(variance)
        self.kernels = self.kernels.astype(np.float32)

        if self.use_bias:
            self.biases = np.zeros(shape=(self.filters, 1, 1, 1), dtype=np.float32)

    def count_params(self) -> int:
        total = np.prod(self.kernels.shape)

        if self.use_bias:
            return total + self.filters

        return total

    def build(self) -> Any:
        self.init_params()

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

        convolutions, self._vec_ip, self._vec_kernel = convolve2d(
            X=self.input(),
            kernel=self.kernels,
            stride=self.stride,
            padding=(self.p_H, self.p_W),
            return_vec_ip=True,
            return_vec_kernel=True,
        )

        if self.use_bias:
            convolutions += self.biases

        activations = (
            self.activation.forward_step(ip=convolutions)
            if self.activation is not None
            else convolutions
        )

        self.convolutions, self.activations = convolutions, activations

        return self.activations

    def _compute_dW(self, dZ: np.ndarray) -> np.ndarray:
        ip = self._vec_ip

        dW = np.matmul(ip[..., None], dZ[..., None, :], dtype=np.float32).sum(
            axis=(0, 1)
        )

        return dW.reshape(-1, self.kernel_H, self.kernel_W, self.filters) / ip.shape[0]

    def backprop_step(self, dA: np.ndarray, *args, **kwargs) -> np.ndarray:
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

        ipH, ipW = self.input_shape()[1:-1]
        padded_shape = (ipH + 2 * self.p_H, ipW + 2 * self.p_W)

        dX = accumulate_dX_conv(
            dX_shape=(dZ.shape[0], self.ip_C, *padded_shape),
            output_size=self.output_area(),
            dIp=np.matmul(dZ, self._vec_kernel.T, dtype=np.float32),
            stride=self.stride,
            kernel_size=self.kernel_size,
            reshape=(-1, self.ip_C, self.kernel_H, self.kernel_W),
            padding=(self.p_H, self.p_W),
        )

        self.reset_attrs()

        return dX
