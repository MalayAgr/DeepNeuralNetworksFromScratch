from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np
from numba import njit


@njit(cache=True)
def _clip(a: np.ndarray, epsilon: float) -> np.ndarray:
    if ((a == 1) | (a <= 0)).any():
        a = np.maximum(a, epsilon).astype(np.float32)
        a = np.minimum(1 - epsilon, a).astype(np.float32)
    return a


@njit(cache=True)
def _binary_crossentropy(
    labels: np.ndarray, preds: np.ndarray, epsilon: float
) -> float:
    preds = _clip(preds, epsilon)
    loss = labels * np.log(preds)
    loss += (1 - labels) * np.log(1 - preds)
    loss = np.sum(-loss)
    loss /= labels.shape[-1]
    return loss


@njit(cache=True)
def _binary_crossentropy_derivative(
    labels: np.ndarray, preds: np.ndarray, epsilon: float
) -> np.ndarray:
    preds = _clip(preds, epsilon)
    grad = 1 - labels
    grad /= 1 - preds
    grad -= labels / preds
    grad /= labels.shape[-1]
    return grad


@njit(cache=True)
def _categorical_crossentropy(labels: np.ndarray, preds: np.ndarray) -> float:
    prod = labels * np.log(preds)
    bs = labels.shape[-1]
    loss = 0.0
    for idx in np.arange(bs):
        loss += -prod[..., idx].sum()
    loss /= bs
    return loss


@njit(cache=True)
def _categorical_crossentropy_derivative(
    labels: np.ndarray, preds: np.ndarray
) -> np.ndarray:
    grad = -labels
    grad /= preds
    grad /= labels.shape[-1]
    return grad


class Loss(ABC):
    names: list[str] = None
    REGISTRY: dict[str, type[Loss]] = {}
    ndim: int = None

    def __init_subclass__(cls, **kwargs) -> None:
        if (names := cls.names) is not None:
            Loss.REGISTRY.update({name: cls for name in names})

    def __str__(self) -> str:
        return f"{self.__class__.__name__}()"

    def __repr__(self) -> str:
        return self.__str__()

    def validate_input(self, labels: np.ndarray, preds: np.ndarray) -> None:
        if labels.shape != preds.shape:
            raise AttributeError(
                "The labels and the predictions should have the same shape"
            )

        if labels.ndim < self.ndim:
            raise AttributeError(
                f"{self.__class__.__name__} expects at least {self.ndim}-dimensional inputs"
            )

    def should_reshape(self, shape: tuple[int, ...]) -> bool:
        """Method to determine if the labels and predictions should be reshaped."""
        return False

    @staticmethod
    def reshape_labels_and_preds(
        labels: np.ndarray, preds: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        """Method to reshape the labels and predictions if they should be reshaped."""
        return labels, preds

    @abstractmethod
    def loss_func(self, labels: np.ndarray, preds: np.ndarray) -> float:
        """
        The formula used to calculate the loss.
        Subclasses classes must implement this.

        If the loss is J with inputs preds and Y,
        this should return J(preds, Y).

        Arguments:
            preds: Numpy-array, the predictions to be used for calculating the loss.

        Returns:
            A float representing the loss.
        """

    @abstractmethod
    def loss_derivative(self, labels: np.ndarray, preds: np.ndarray) -> np.ndarray:
        """
        The formula used to calculate the derivative of the loss function
        With respect to preds.
        Subclasses classes must implement this.

        If the loss is J with inputs preds and Y,
        this should return J'(preds, Y).

        Arguments:
            preds: Numpy-array, the predictions to be used for calculating the derivatives.

        Returns:
            A Numpy-array with the calculated derivatives.
        """

    def compute_loss(self, labels: np.ndarray, preds: np.ndarray) -> float:
        self.validate_input(labels, preds)

        if self.should_reshape(labels.shape):
            labels, preds = self.reshape_labels_and_preds(labels, preds)

        return self.loss_func(labels, preds)

    def compute_derivatives(self, labels: np.ndarray, preds: np.ndarray) -> np.ndarray:
        self.validate_input(labels, preds)

        old_shape = None

        if self.should_reshape(labels.shape):
            old_shape = labels.shape
            labels, preds = self.reshape_labels_and_preds(labels, preds)

        grad = self.loss_derivative(labels, preds).astype(np.float32)

        if old_shape is not None:
            grad.shape = old_shape

        return grad


class BinaryCrossEntropy(Loss):
    names = ["binary_crossentropy", "bce"]
    ndim = 2
    epsilon = 1e-15

    def should_reshape(self, shape: tuple[int, ...]) -> bool:
        return len(shape) > self.ndim or shape[0] != 1

    @staticmethod
    def reshape_labels_and_preds(
        labels: np.ndarray, preds: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        return labels.reshape(1, -1), preds.reshape(1, -1)

    def loss_func(self, labels: np.ndarray, preds: np.ndarray) -> float:
        return _binary_crossentropy(labels=labels, preds=preds, epsilon=self.epsilon)

    def loss_derivative(self, labels: np.ndarray, preds: np.ndarray) -> np.ndarray:
        return _binary_crossentropy_derivative(
            labels=labels,
            preds=preds,
            epsilon=self.epsilon,
        )


class MeanSquaredError(Loss):
    names = ["mean_squared_error", "mse"]
    ndim = 2

    def should_reshape(self, shape: tuple) -> bool:
        return len(shape) > self.ndim or shape[0] != 1

    @staticmethod
    def reshape_labels_and_preds(
        labels: np.ndarray, preds: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        return labels.reshape(1, -1), preds.reshape(1, -1)

    def loss_func(self, labels: np.ndarray, preds: np.ndarray) -> float:
        loss = preds - labels
        loss **= 2
        loss = np.sum(loss / labels.shape[-1])

        return np.squeeze(loss)

    def loss_derivative(self, labels: np.ndarray, preds: np.ndarray) -> np.ndarray:
        grad = preds - labels
        grad *= 2
        grad /= labels.shape[-1]

        return grad


class CategoricalCrossEntropy(Loss):
    names = ["categorial_crossentropy", "cce"]
    ndim = 2

    def should_reshape(self, shape: tuple) -> bool:
        return len(shape) > self.ndim

    @staticmethod
    def reshape_labels_and_preds(
        labels: np.ndarray, preds: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        classes = labels.shape[0]
        return labels.reshape(classes, -1), preds.reshape(classes, -1)

    def loss_func(self, labels: np.ndarray, preds: np.ndarray) -> float:
        return _categorical_crossentropy(labels=labels, preds=preds)

    def loss_derivative(self, labels: np.ndarray, preds: np.ndarray) -> np.ndarray:
        return _categorical_crossentropy_derivative(labels=labels, preds=preds)
