from __future__ import annotations

import numpy as np
from dnn.utils import StateVariable

from ..schedulers import LearningRateType
from .base_optimizer import Optimizer, WeightsGradientsType


class RMSProp(Optimizer):
    rho = StateVariable()
    epsilon = StateVariable()

    def __init__(
        self,
        learning_rate: LearningRateType = 1e-2,
        rho: float = 0.9,
        epsilon: float = 1e-7,
    ) -> None:

        if not 0.0 <= rho <= 1.0:
            raise ValueError("rho should be between 0 and 1.")

        super().__init__(learning_rate=learning_rate)

        self.epsilon = epsilon

        self.rho = rho

        self._root_mean_sqs: list[np.ndarray] = None

    def pre_iteration_state(self, grads: WeightsGradientsType) -> None:
        super().pre_iteration_state(grads=grads)

        if self.iterations == 0:
            self._root_mean_sqs = [np.zeros_like(weight) for weight, _ in grads]

    def _update_rms(self, grad: np.ndarray, rms: np.ndarray) -> np.ndarray:
        rho = self.rho
        one_minus_rho = 1 - rho

        rms *= rho
        rms += one_minus_rho * np.square(grad)

        return rms

    def _apply_gradient(
        self, weight: np.ndarray, gradient: np.ndarray, grad_idx: int
    ) -> None:
        rms = self._root_mean_sqs[grad_idx]
        rms = self._update_rms(gradient, rms)

        weight -= self.lr * gradient / (np.sqrt(rms) + self.epsilon)
