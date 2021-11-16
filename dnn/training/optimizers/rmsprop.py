from typing import List, Tuple, Union

import numpy as np

from dnn.training.schedulers import LearningRateScheduler

from .base_optimizer import Optimizer


class RMSProp(Optimizer):
    def __init__(
        self,
        learning_rate: Union[float, LearningRateScheduler] = 1e-2,
        rho=0.9,
        epsilon: float = 1e-7,
    ) -> None:

        if not 0.0 <= rho <= 1.0:
            raise ValueError("rho should be between 0 and 1.")

        super().__init__(learning_rate=learning_rate)

        self.add_or_update_state_variable("epsilon", epsilon)
        self.add_or_update_state_variable("rho", rho)

    @property
    def rho(self):
        return self.fetch_state_variable("rho")

    @property
    def epsilon(self):
        return self.fetch_state_variable("epsilon")

    def pre_iteration_state(self, grads: List[Tuple[np.ndarray, np.ndarray]]) -> None:
        super().pre_iteration_state(grads=grads)

        if self.iterations == 0:
            root_mean_sqs = [np.zeros_like(weight) for weight, _ in grads]
            self.add_or_update_state_variable("root_mean_sqs", root_mean_sqs)

    def _update_rms(self, grad: np.ndarray, rms: np.ndarray) -> np.ndarray:
        rho = self.rho
        one_minus_rho = 1 - rho

        rms *= rho
        rms += one_minus_rho * np.square(grad)

        return rms

    def _apply_gradient(
        self, weight: np.ndarray, gradient: np.ndarray, grad_idx: int
    ) -> None:
        root_mean_sqs = self.fetch_state_variable("root_mean_sqs")

        rms = root_mean_sqs[grad_idx]
        rms = self._update_rms(gradient, rms)

        weight -= self.lr * gradient / (np.sqrt(rms) + self.epsilon)
