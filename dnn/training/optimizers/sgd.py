from typing import List, Tuple

import numpy as np
from .base_optimizer import Optimizer


class SGD(Optimizer):
    def __init__(
        self,
        learning_rate: float = 1e-2,
        momentum=0.0,
    ) -> None:

        super().__init__(learning_rate=learning_rate)

        if not 0.0 <= momentum <= 1.0:
            raise ValueError("momentum should be between 0 and 1.")

        self._momentum = momentum > 0.0

        self.add_or_update_state_variable("momentum", momentum)

    @property
    def momentum(self):
        return self.fetch_state_variable("momentum")

    def _update_velocity(self, grad: np.ndarray, velocity: np.ndarray) -> np.ndarray:
        mom = self.momentum
        one_minus_mom = 1 - mom

        velocity *= mom
        velocity += one_minus_mom * grad

        return velocity

    def pre_iteration_state(self, grads: List[Tuple[np.ndarray, np.ndarray]]) -> None:
        super().pre_iteration_state(grads=grads)

        if self._momentum is True and self.iterations == 0:
            velocities = [np.zeros_like(weight) for weight, _ in grads]
            self.add_or_update_state_variable("velocities", velocities)

    def _apply_gradient(
        self, weight: np.ndarray, gradient: np.ndarray, grad_idx: int
    ) -> None:

        lr = self.lr
        update = gradient

        if self._momentum:
            velocities = self.fetch_state_variable("velocities")

            velocity = velocities[grad_idx]
            update = self._update_velocity(gradient, velocity)

        weight -= lr * update
