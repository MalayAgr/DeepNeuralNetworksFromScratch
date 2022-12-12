from __future__ import annotations

import numpy as np
from dnn.utils import StateVariable

from ..schedulers import LearningRateType
from .base_optimizer import Optimizer, StateVariable, WeightsGradientsType


class SGD(Optimizer):
    momentum = StateVariable()

    def __init__(
        self, learning_rate: LearningRateType = 1e-2, momentum: float = 0.0
    ) -> None:

        super().__init__(learning_rate=learning_rate)

        if not 0.0 <= momentum <= 1.0:
            raise ValueError("momentum should be between 0 and 1.")

        self._momentum = momentum > 0.0

        self.momentum = momentum

        self._velocities: list[np.ndarray] = None

    def _update_velocity(self, grad: np.ndarray, velocity: np.ndarray) -> np.ndarray:
        mom = self.momentum
        one_minus_mom = 1 - mom

        velocity *= mom
        velocity += one_minus_mom * grad

        return velocity

    def pre_iteration_state(self, grads: WeightsGradientsType) -> None:
        super().pre_iteration_state(grads=grads)

        if self._momentum is True and self.iterations == 0:
            self._velocities = [np.zeros_like(weight) for weight, _ in grads]

    def _apply_gradient(
        self, weight: np.ndarray, gradient: np.ndarray, grad_idx: int
    ) -> None:

        update = gradient

        if self._momentum:
            velocity = self._velocities[grad_idx]
            update = self._update_velocity(gradient, velocity)

        weight -= self.lr * update
