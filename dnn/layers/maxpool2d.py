from __future__ import annotations

from typing import Tuple

import numpy as np

from .base_layer_pooling import BasePooling
from .utils import conv_utils as cutils


class MaxPooling2D(BasePooling):
    def pool_func(self, X: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        return cutils.maxpool2D(
            X=X, windows=self.windows, output_size=self.output_area()
        )
