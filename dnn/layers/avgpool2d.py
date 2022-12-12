from __future__ import annotations

import numpy as np

from .base_layer_pooling import BasePooling
from .utils import conv_utils as cutils


class AveragePooling2D(BasePooling):
    def pool_func(self, X: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        return cutils.averagepool2D(
            X=X, windows=self.windows, output_size=self.output_area()
        )
