from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np
from dnn.utils import StateVariable

from ..schedulers import LearningRateType

WeightsGradientsType = list[tuple[np.ndarray, np.ndarray]]


class Optimizer(ABC):
    iterations = StateVariable()
    scheduler = StateVariable()
    lr = StateVariable()

    def __init__(self, learning_rate: LearningRateType = 1e-2) -> None:
        self.state = {}
        self.iterations = 0

        if not isinstance(learning_rate, float):
            self.scheduler = learning_rate
            self.lr = 0.0
        else:
            self.lr = learning_rate

    def __str__(self) -> str:
        attrs = ", ".join(f"{name}={value}" for name, value in self.state.items())
        return f"{self.__class__.__name__}({attrs})"

    def __repr__(self) -> str:
        return self.__str__()

    @property
    def state_variables(self) -> list[str]:
        """The state variables in the optimizer."""
        return list(self.state.keys())

    def pre_iteration_state(self, grads: WeightsGradientsType) -> None:
        """
        Method to prepare the state of the optimizer before a minimization iteration.

        By default, it computes the new learning rate if a scheduler is being used.

        If child classes override this method, they MUST call super()
        to ensure the optimizer works as expected.
        """
        if self.scheduler is not None:
            self.lr = self.scheduler.lr(self.iterations)

    def post_iteration_state(self, grads: WeightsGradientsType) -> None:
        """
        Method to update the state of the optimizer after a minimization iteration.

        By default, it increments the iteration count.
        """
        self.iterations += 1

    def minimize(self, weights_and_grads: WeightsGradientsType) -> None:
        self.pre_iteration_state(grads=weights_and_grads)

        self.apply_gradients(grads=weights_and_grads)

        self.post_iteration_state(grads=weights_and_grads)

    @abstractmethod
    def _apply_gradient(
        self, weight: np.ndarray, gradient: np.ndarray, grad_idx: int
    ) -> None:
        """
        Method to apply the gradient to a single weight.

        The update MUST be done IN-PLACE on the weight to ensure the update
        has an effect.
        """

    def apply_gradients(self, grads: WeightsGradientsType):
        for idx, (weight, grad) in enumerate(grads):
            self._apply_gradient(weight, grad, idx)
