from __future__ import annotations

from typing import List, Tuple, Union

import numpy as np
from numba import njit

from dnn.training.schedulers import LearningRateScheduler

from .base_optimizer import Optimizer, WeightsGradientsType


@njit(cache=True, parallel=True)
def _update_first_moment(
    moment: np.ndarray, grad: np.ndarray, beta1: float
) -> np.ndarray:
    moment = np.multiply(moment, beta1)
    moment = np.add(moment, np.multiply(grad, 1 - beta1))
    return moment


@njit(cache=True, parallel=True)
def _update_second_moment(
    moment: np.ndarray, grad: np.ndarray, beta2: float
) -> np.ndarray:
    grad = np.square(grad)
    moment = np.multiply(moment, beta2)
    moment = np.add(moment, np.multiply(grad, 1 - beta2))
    return moment


@njit(cache=True, parallel=True)
def _maximum(
    x: Union[np.ndarray, float], y: Union[np.ndarray, float]
) -> Union[np.ndarray, float]:
    return np.maximum(x, y)


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
        self._first_moments: List[np.ndarray] = None
        self._second_moments: List[np.ndarray] = None

    @property
    def beta_1(self):
        return self.fetch_state_variable("beta_1")

    @property
    def beta_2(self):
        return self.fetch_state_variable("beta_2")

    @property
    def epsilon(self):
        return self.fetch_state_variable("epsilon")

    def pre_iteration_state(self, grads: WeightsGradientsType) -> None:
        super().pre_iteration_state(grads)

        if self.iterations == 0:
            first_moments, second_moments = [], []
            for weight, _ in grads:
                first_moments.append(np.zeros_like(weight))
                second_moments.append(np.zeros_like(weight))

            self._first_moments, self._second_moments = first_moments, second_moments

    def _compute_update(self, grad: np.ndarray, idx: int) -> np.ndarray:
        first_moms = self._first_moments
        second_moms = self._second_moments

        m1 = _update_first_moment(first_moms[idx], grad, self.beta_1)
        first_moms[idx] = m1

        current_m2 = second_moms[idx]
        m2 = _update_second_moment(current_m2, grad, self.beta_2)
        second_moms[idx] = m2

        if self.amsgrad:
            m2 = _maximum(m2, current_m2)

        if self.bias_correction:
            m1 = np.divide(m1, 1 - self._beta1t)
            m2 = np.divide(m2, 1 - self._beta2t)

            self._beta1t *= self.beta_1
            self._beta2t *= self.beta_2

        return m1 / (np.sqrt(m2) + self.epsilon)

    def _apply_gradient(
        self, weight: np.ndarray, gradient: np.ndarray, grad_idx: int
    ) -> None:
        update = self._compute_update(gradient, grad_idx)

        weight -= self.lr * update
