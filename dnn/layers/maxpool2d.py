from __future__ import annotations

from typing import Tuple

import numpy as np

from .base_pooling import BasePooling
from .utils import conv_utils as cutils


class MaxPooling2D(BasePooling):
    reset = ("pooled", "_gradient_mask")

    str_attrs = ("pool_size", "stride", "padding")

    __slots__ = ("pooled", "_gradient_mask")

    def pool_func(self, X: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        return cutils.maxpool2D(
            X=X, windows=self.windows, output_size=self.output_area()
        )
