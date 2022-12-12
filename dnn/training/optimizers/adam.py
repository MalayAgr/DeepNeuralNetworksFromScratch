from __future__ import annotations

import numpy as np
from dnn.utils import StateVariable
from numba import njit

from ..schedulers import LearningRateType
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
def _maximum(x: np.ndarray | float, y: np.ndarray | float) -> np.ndarray | float:
    return np.maximum(x, y)


@njit(cache=True, parallel=True)
def _compute_update(
    m1: np.ndarray, m2: np.ndarray, epsilon: float, lr: float
) -> np.ndarray:
    m2 = m2 ** (1.0 / 2)
    m2 = m2 + epsilon
    m1 = m1 / m2
    return lr * m1


class Adam(Optimizer):
    beta_1 = StateVariable()
    beta_2 = StateVariable()
    epsilon = StateVariable()

    def __init__(
        self,
        learning_rate: LearningRateType = 1e-2,
        beta_1: float = 0.9,
        beta_2: float = 0.999,
        epsilon: float = 1e-8,
        amsgrad: bool = False,
        bias_correction: bool = True,
    ) -> None:
        super().__init__(learning_rate=learning_rate)

        self.beta_1 = beta_1

        self.beta_2 = beta_2

        self.epsilon = epsilon

        self.amsgrad = amsgrad
        self.bias_correction = bias_correction

        self._beta1t = beta_1
        self._beta2t = beta_2
        self._first_moments: list[np.ndarray] = None
        self._second_moments: list[np.ndarray] = None

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

        return _compute_update(m1, m2, self.epsilon, self.lr)

    def _apply_gradient(
        self, weight: np.ndarray, gradient: np.ndarray, grad_idx: int
    ) -> None:
        weight -= self._compute_update(gradient, grad_idx)
