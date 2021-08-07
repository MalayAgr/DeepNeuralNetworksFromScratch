from typing import Tuple, Union

import numpy as np

from .base_conv import Conv
from .utils import (
    backprop_bias_conv,
    backprop_ip_conv2d,
    backprop_kernel_conv2d,
    convolve2d,
)


class Conv2D(Conv):
    def conv_func(
        self,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        return convolve2d(
            X=self.input(),
            kernel=self.kernels,
            stride=self.stride,
            padding=(self.p_H, self.p_W),
            return_vec_ip=True,
            return_vec_kernel=True,
        )

    def _reshape_dZ(self, dZ: np.ndarray) -> np.ndarray:
        return np.swapaxes(dZ, 0, -1).reshape(dZ.shape[-1], -1, self.filters)

    def _compute_dW(self, dZ: np.ndarray) -> np.ndarray:
        return backprop_kernel_conv2d(
            ip=self._vec_ip, grad=dZ, kernel_size=self.kernel_size, filters=self.filters
        )

    def _compute_dB(self, dZ: np.ndarray) -> np.ndarray:
        return backprop_bias_conv(grad=dZ, axis=(0, 1), reshape=self.biases.shape[1:])

    def _compute_dVec_Ip(self, dZ: np.ndarray) -> np.ndarray:
        return backprop_ip_conv2d(grad=dZ, kernel=self._vec_kernel)
