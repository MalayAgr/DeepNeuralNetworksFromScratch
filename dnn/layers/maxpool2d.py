from __future__ import annotations

from typing import Optional, Tuple

import numpy as np

from .base_layer import BaseLayer, LayerInput
from .utils import conv_utils as cutils


class MaxPooling2D(BaseLayer):
    reset = ("pooled", "_dX_share")

    str_attrs = ("pool_size", "stride", "padding")

    __slots__ = ("pooled", "_dX_share")

    def __init__(
        self,
        ip: LayerInput,
        pool_size: Tuple,
        stride: Tuple[int, int] = (2, 2),
        padding: str = "valid",
        name: str = None,
    ) -> None:
        self.pool_size = pool_size
        self.pool_H, self.pool_W = pool_size

        self.stride = stride
        self.stride_H, self.stride_W = stride

        self.padding = padding
        self.p_H, self.p_W = cutils.compute_conv_padding(pool_size, mode=padding)

        super().__init__(ip, trainable=False, name=name)

        self.windows = self.input_shape()[0]

        self.pooled = None
        self._dX_share = None

    def fans(self) -> Tuple[int, int]:
        return self.ip_C, self.ip_C

    def output(self) -> Optional[np.ndarray]:
        return self.pooled

    def output_area(self) -> Tuple[int, int]:
        ip_shape = self.input_shape()
        ipH, ipW = ip_shape[1], ip_shape[2]

        oH = cutils.compute_conv_output_dim(ipH, self.pool_H, self.p_H, self.stride_H)
        oW = cutils.compute_conv_output_dim(ipW, self.pool_W, self.p_W, self.stride_W)

        return oH, oW

    def output_shape(self) -> Tuple:
        if self.pooled is not None:
            return self.pooled.shape

        oH, oW = self.output_area()

        return self.windows, oH, oW, None

    def _get_pool_outputs(self, ip: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
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

        shape = (self.windows, *self.output_area(), -1)

        return np.swapaxes(maximums, 0, -1).reshape(*shape), mask

    def _pool(self, X: np.ndarray) -> np.ndarray:
        X = cutils.pad(X, self.p_H, self.p_W)

        X = cutils.vectorize_ip_for_conv(
            X=X,
            kernel_size=self.pool_size,
            stride=self.stride,
            output_size=self.output_area(),
            reshape=(self.windows, self.pool_H * self.pool_W, X.shape[-1]),
        )

        X = np.moveaxis(X, -1, 0)

        pooled, self._dX_share = self._get_pool_outputs(ip=X)

        return pooled

    def forward_step(self, *args, **kwargs) -> np.ndarray:
        self.pooled = self._pool(self.input())
        return self.pooled

    def transform_backprop_gradient(
        self, grad: np.ndarray, *args, **kwargs
    ) -> np.ndarray:
        return np.swapaxes(grad, 0, -1).reshape(grad.shape[-1], -1, self.windows)

    def backprop_inputs(self, grad: np.ndarray, *args, **kwargs) -> np.ndarray:
        ipH, ipW = self.input_shape()[1:-1]
        padded_shape = (ipH + 2 * self.p_H, ipW + 2 * self.p_W)

        return cutils.accumulate_dX_conv(
            dX_shape=(grad.shape[0], self.windows, *padded_shape),
            output_size=self.output_area(),
            dIp=self._dX_share * grad[..., None],
            stride=self.stride,
            kernel_size=self.pool_size,
            reshape=(-1, self.windows, self.pool_H, self.pool_W),
            padding=(self.p_H, self.p_W),
        )
