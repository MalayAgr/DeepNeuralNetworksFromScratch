from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Tuple, Union

import numpy as np

from dnn.training.schedulers import LearningRateScheduler, LearningRateType

WeightsGradientsType = List[Tuple[np.ndarray, np.ndarray]]


class Optimizer(ABC):
    def __init__(self, learning_rate: LearningRateType = 1e-2) -> None:
        self._state = {}
        self._state["iterations"] = 0
        self._state["lr"] = learning_rate
        self._scheduler = isinstance(learning_rate, LearningRateScheduler)
        self._state["lr_t"] = None if self._scheduler else learning_rate

    def __str__(self) -> str:
        return f"{self.__class__.__name__}()"

    def __repr__(self) -> str:
        return self.__str__()

    @property
    def state(self) -> Dict:
        """The current state of the optimizer."""
        return self._state

    @property
    def state_variables(self) -> List[str]:
        """The state variables in the optimizer."""
        return list(self._state.keys())

    @property
    def lr(self) -> float:
        """The current learing rate of the optimizer."""
        return self._state["lr_t"]

    @property
    def iterations(self) -> int:
        """The number of minimization iterations run so far."""
        return self._state["iterations"]

    def fetch_state_variable(self, state_var: str) -> Any:
        return self._state[state_var]

    def add_or_update_state_variable(self, state_var: str, value: Any) -> None:
        """Method to add a new state variable."""
        self._state[state_var] = value

    def pre_iteration_state(self, grads: WeightsGradientsType) -> None:
        """
        Method to prepare the state of the optimizer before a minimization iteration.

        By default, it computes the new learning rate if a scheduler is being used.

        If child classes override this method, they MUST call super()
        to ensure the optimizer works as expected.
        """
        if self._scheduler is True:
            scheduler: LearningRateScheduler = self._state["lr"]
            self._state["lr_t"] = scheduler.lr(self.iterations)

    def post_iteration_state(self, grads: WeightsGradientsType) -> None:
        """
        Method to update the state of the optimizer after a minimization iteration.

        By default, it increments the iteration count.
        """
        self._state["iterations"] += 1

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
