from __future__ import annotations

from abc import abstractmethod

import numpy as np
from dnn.utils import HeightWidthAttribute

from .base_layer import BaseLayer, LayerInputType
from .utils import conv_utils as cutils


class BasePooling(BaseLayer):
    reset = ("pooled", "gradient_mask")

    str_attrs = ("pool_size", "stride", "padding")

    __slots__ = ("pooled", "gradient_mask")

    pool_size = HeightWidthAttribute("pool_H", "pool_W")
    stride = HeightWidthAttribute()

    def __init__(
        self,
        ip: LayerInputType,
        pool_size: tuple[int, ...],
        stride: tuple[int, int] = (2, 2),
        padding: str = "valid",
        name: str = None,
    ) -> None:
        self.pool_size = pool_size

        self.stride = stride

        self.padding = padding

        super().__init__(ip, trainable=False, name=name)

        self.windows = self.input_shape()[0]

        self.pooled = None
        self.gradient_mask = None

    def fans(self) -> tuple[int, int]:
        return self.ip_C, self.ip_C

    def output(self) -> np.ndarray | None:
        return self.pooled

    def output_area(self) -> tuple[int, int]:
        ip_shape = self.input_shape()
        ipH, ipW = ip_shape[1], ip_shape[2]

        pH, pW = self.pad_area()
        oH = cutils.convolution_output_dim(ipH, self.pool_H, pH, self.stride_H)
        oW = cutils.convolution_output_dim(ipW, self.pool_W, pW, self.stride_W)

        return oH, oW

    def pad_area(self) -> tuple[int, int]:
        return cutils.padding(self.pool_size, mode=self.padding)

    def output_shape(self) -> tuple[int, ...]:
        if self.pooled is not None:
            return self.pooled.shape

        oH, oW = self.output_area()

        return self.windows, oH, oW, None

    def _padded_shape(self) -> tuple[int, int]:
        ipH, ipW = self.input_shape()[1:-1]
        pH, pW = self.pad_area()
        return ipH + 2 * pH, ipW + 2 * pW

    @abstractmethod
    def pool_func(self, X: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        pass

    def _pool(self, X: np.ndarray) -> np.ndarray:
        X = cutils.prepare_ip(
            X=X,
            kernel_size=self.pool_size,
            stride=self.stride,
            padding=self.pad_area(),
            vec_reshape=(self.windows, self.pool_H * self.pool_W, -1),
        )
        X = np.moveaxis(X, -1, 0)

        pooled, self.gradient_mask = self.pool_func(X)

        return pooled

    def forward_step(self, *args, **kwargs) -> np.ndarray:
        self.pooled = self._pool(self.input())
        return self.pooled

    def transform_backprop_gradient(
        self, grad: np.ndarray, *args, **kwargs
    ) -> np.ndarray:
        return np.swapaxes(grad, 0, -1).reshape(grad.shape[-1], -1, self.windows)

    def backprop_inputs(self, grad: np.ndarray, *args, **kwargs) -> np.ndarray:
        post_pad_H, post_pad_W = self._padded_shape()
        grad_shape = (grad.shape[0], self.windows, post_pad_H, post_pad_W)
        reshape = (-1, self.windows, self.pool_H, self.pool_W)

        return cutils.accumulate_dX_conv(
            grad_shape=grad_shape,
            output_size=self.output_area(),
            vec_ip_grad=self.gradient_mask * grad[..., None],
            stride=self.stride,
            kernel_size=self.pool_size,
            reshape=reshape,
            padding=self.pad_area(),
        )
