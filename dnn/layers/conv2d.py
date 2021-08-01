from typing import Tuple, Union

import numpy as np

from .base_conv import Conv
from .utils import convolve2d


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
        ip = self._vec_ip

        dW = np.matmul(ip[..., None], dZ[..., None, :], dtype=np.float32).sum(
            axis=(0, 1)
        )

        return dW.reshape(-1, self.kernel_H, self.kernel_W, self.filters)

    def _compute_dB(self, dZ: np.ndarray) -> np.ndarray:
        return dZ.sum(axis=(0, 1)).reshape(-1, *self.biases.shape[1:])

    def _compute_dVec_Ip(self, dZ: np.ndarray) -> np.ndarray:
        return np.matmul(dZ, self._vec_kernel.T, dtype=np.float32)
