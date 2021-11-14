from typing import List, Tuple, Union

import numpy as np

from dnn.training.schedulers import LearningRateScheduler

from .base_optimizer import Optimizer


class Adam(Optimizer):
    def __init__(
        self,
        learning_rate: Union[float, LearningRateScheduler] = 1e-2,
        beta_1: float = 0.9,
        beta_2: float = 0.999,
        epsilon: float = 1e-8,
        amsgrad: bool = False,
        bias_correction: bool = True,
    ) -> None:
        super().__init__(learning_rate=learning_rate)

        self.add_or_update_state_variable("beta_1", beta_1)
        self.add_or_update_state_variable("beta_2", beta_2)
        self.add_or_update_state_variable("epsilon", epsilon)

        self.amsgrad = amsgrad
        self.bias_correction = bias_correction

        self._beta1t = beta_1
        self._beta2t = beta_2

    @property
    def beta_1(self):
        return self.fetch_state_variable("beta_1")

    @property
    def beta_2(self):
        return self.fetch_state_variable("beta_2")

    @property
    def epsilon(self):
        return self.fetch_state_variable("epsilon")

    def pre_iteration_state(self, grads: List[Tuple[np.ndarray, np.ndarray]]) -> None:
        super().pre_iteration_state(grads)

        if self.iterations == 0:
            first_moments, second_moments = [], []
            for weight, _ in grads:
                first_moments.append(np.zeros_like(weight))
                second_moments.append(np.zeros_like(weight))

            self.add_or_update_state_variable("first_moments", first_moments)
            self.add_or_update_state_variable("second_moments", second_moments)

    def _update_first_moment(self, grad: np.ndarray, moment: np.ndarray) -> np.ndarray:
        beta_1 = self.beta_1
        one_minus = 1 - beta_1

        moment *= beta_1
        moment += one_minus * grad

        return moment

    def _update_second_moment(
        self, grad: np.ndarray, moment: np.ndarray, make_copy: bool = False
    ) -> Tuple[np.ndarray, Union[np.ndarray, None]]:

        copy = moment.copy() if make_copy else None

        beta_2 = self.beta_2
        one_minus = 1 - beta_2

        moment *= beta_2
        moment += one_minus * np.square(grad)

        return moment, copy

    def _compute_update(
        self, grad: np.ndarray, m1: np.ndarray, m2: np.ndarray
    ) -> np.ndarray:

        m1 = self._update_first_moment(grad, m1)

        m2, m2_copy = self._update_second_moment(grad, m2, make_copy=self.amsgrad)

        if self.amsgrad:
            m2 = np.maximum(m2, m2_copy)
            m2_copy = None

        if self.bias_correction:
            m1 = np.divide(m1, 1 - self._beta1t)
            m2 = np.divide(m2, 1 - self._beta2t)

            self._beta1t *= self.beta_1
            self._beta2t *= self.beta_2

        return m1 / (np.sqrt(m2) + self.epsilon)

    def _apply_gradient(
        self, weight: np.ndarray, gradient: np.ndarray, grad_idx: int
    ) -> None:
        first_moms = self.fetch_state_variable("first_moments")
        second_moms = self.fetch_state_variable("second_moments")

        update = self._compute_update(
            gradient, first_moms[grad_idx], second_moms[grad_idx]
        )

        weight -= self.lr * update
