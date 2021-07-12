from dnn.utils import pad
from functools import cached_property
from dnn.layers.base_layer import BaseLayer
from dnn.layers.utils import (
    pad,
    compute_conv_padding,
    compute_conv_output_dim,
    vectorize_for_conv,
    accumulate_dX_conv,
)

import numpy as np


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
