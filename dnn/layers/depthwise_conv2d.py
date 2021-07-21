import numpy as np
from dnn.layers.conv2d import Conv2D
from dnn.layers.activations import Activation
from typing import Any, Optional, Tuple, Union

from .base_layer import BaseLayer, LayerInput
from .utils import (
    accumulate_dX_conv,
    depthwise_convolve2d,
)


class DepthwiseConv2D(Conv2D):
    str_attrs = Conv2D.str_attrs[1:]

    def __init__(
        self,
        ip: LayerInput,
        *args,
        kernel_size: Tuple[int, int],
        stride: Tuple[int, int] = (1, 1),
        activation: Optional[Union[Activation, str]] = None,
        multiplier: int = 1,
        padding: str = "valid",
        initializer: str = "he",
        use_bias: bool = True,
        **kwargs
    ) -> None:
        super().__init__(
            ip=ip,
            filters=multiplier,
            kernel_size=kernel_size,
            stride=stride,
            activation=activation,
            padding=padding,
            initializer=initializer,
            use_bias=use_bias,
        )

        self.multiplier = multiplier

    def fans(self) -> Tuple[int, int]:
        fan_in, fan_out = super().fans()

        return fan_in, fan_out * self.ip_C

    def build(self) -> Any:
        super().build()

        if self.use_bias:
            remaining = (self.filters - 1) * self.ip_C
            extra = self._add_param(shape=(remaining, 1, 1, 1), initializer="zeros")
            self.biases = np.concatenate(self.biases, extra)

    def output_shape(self) -> Tuple:
        shape = super().output_shape()

        if self.activations is not None:
            return shape

        c, h, w, m = shape

        return c * self.ip_C, h, w, m

    def forward_step(self, *args, **kwargs) -> np.ndarray:
        self.convolutions, self._vec_ip, self._vec_kernel = depthwise_convolve2d(
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

    def _compute_dW(self, dZ: np.ndarray) -> np.ndarray:
        ip = np.swapaxes(self._vec_ip, -1, -2)

        dZ = np.moveaxis(dZ, 1, 2)

        dW = np.matmul(ip, dZ).sum(axis=0)

        return (
            dW.reshape(-1, self.kernel_H, self.kernel_W, self.multiplier) / ip.shape[0]
        )

    def backprop_step(self, dA: np.ndarray, *args, **kwargs) -> np.ndarray:
        dA = self.activation.backprop_step(dA, ip=self.convolutions)

        dA = np.swapaxes(dA, -1, 0).reshape(
            dA.shape[-1], -1, self.ip_C, self.multiplier
        )

        dW = self._compute_dW(dA)

        m = dA.shape[0]

        reg_param = kwargs.pop("reg_param", 0.0)
        if reg_param > 0:
            dW += (reg_param / m) * self.kernels

        self.gradients["kernels"] = dW

        if self.use_bias:
            self.gradients["biases"] = (
                dA.sum(axis=(0, 1)).reshape(-1, *self.biases.shape[1:]) / m
            )

        if self.requires_dX is False:
            self.reset_attrs()
            return
