from abc import ABC, abstractmethod
from typing import Any, Dict, List, Tuple

import numpy as np

from dnn.training.graph.core import ComputationGraph


class Optimizer(ABC):
    def __init__(self, learning_rate: float = 1e-2) -> None:
        self._state = {}
        self._state["iterations"] = 0
        self._state["lr"] = learning_rate

    @property
    def state(self) -> Dict:
        """The current state of the optimizer."""
        return self._state

    @property
    def state_variables(self) -> List[str]:
        """The state variables in the optimizer."""
        return list(self._state.keys())

    @property
    def learning_rate(self) -> float:
        """The current learing rate of the optimizer."""
        return self._state["lr"]

    @property
    def iterations(self) -> int:
        """The number of minimization iterations run so far."""
        return self._state["iterations"]

    def fetch_state_variable(self, state_var) -> Any:
        """Method to obtain the value of a state variable."""
        return self._state[state_var]

    def add_or_update_state_variable(self, state_var: str, value: Any) -> None:
        """Method to add a new state variable."""
        self._state[state_var] = value

    def pre_iteration_state(self, grads: List[Tuple[np.ndarray, np.ndarray]]) -> None:
        """
        Method to prepare the optimizer before a minimization iteration.

        By default, it does nothing.
        """

    def post_iteration_state(self, grads: List[Tuple[np.ndarray, np.ndarray]]) -> None:
        """
        Method to update the optimizer after a minimization iteration.

        By default, it increments the number of minimization iterations.
        """
        self._state["iterations"] += 1

    def minimize(self, graph: ComputationGraph, initial_grad: np.ndarray):
        weights_and_grads = graph.backprop(grad=initial_grad)

        self.pre_iteration_state(grads=weights_and_grads)

        self.apply_gradients(grads=weights_and_grads)

        self.post_iteration_state(grads=weights_and_grads)


    @abstractmethod
    def _apply_gradient(self, weight: np.ndarray, gradient: np.ndarray, grad_idx: int) -> None:
        """
        Method to apply the gradient to a single weight.

        The update MUST be done IN-PLACE on the weight to ensure the update
        has an effect.
        """

    def apply_gradients(self, grads: List[np.ndarray, np.ndarray]):
        for idx, (weight, grad) in enumerate(grads):
            self._apply_gradient(weight, grad, idx)
