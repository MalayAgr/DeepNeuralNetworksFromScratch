from __future__ import annotations

import numpy as np

from .maxpool2d import MaxPooling2D


class AveragePooling2D(MaxPooling2D):
    def _get_pool_outputs(self, ip: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        ip_shape = ip.shape

        averages = ip.mean(axis=-1)

        distributed = (
            np.ones(shape=(1, 1, 1, ip_shape[-1]), dtype=np.float32) / ip_shape[-1]
        )

        shape = (self.windows, self.out_H, self.out_W, -1)

        return np.swapaxes(averages, 0, -1).reshape(*shape), distributed
