from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Tuple

import numpy as np


class Loss(ABC):
    name = None
    ndim = None

    @classmethod
    def get_loss_classes(cls) -> dict:
        result = {}

        for sub_cls in cls.__subclasses__():
            result.update(sub_cls.get_loss_classes())
            if sub_cls.name is not None:
                result.update({name: sub_cls for name in sub_cls.name})
        return result

    @staticmethod
    def validate_input(labels: np.ndarray, preds: np.ndarray) -> None:
        if labels.shape != preds.shape:
            raise AttributeError(
                "The labels and the predictions should have the same shape"
            )

    @staticmethod
    @abstractmethod
    def reshape_ip(ip: np.ndarray) -> np.ndarray:
        """Method to reshape the labels or predictions if they have ndim > self.ndim"""

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

        if labels.ndim > self.ndim:
            labels = self.reshape_ip(labels)
            preds = self.reshape_ip(preds)

        return self.loss_func(labels, preds).astype(np.float32)

    def compute_derivatives(self, labels: np.ndarray, preds: np.ndarray) -> np.ndarray:
        self.validate_input(labels, preds)

        restore_dims, old_shape = False, None
        old_shape = None

        if labels.ndim > self.ndim:
            restore_dims, old_shape = True, labels.shape
            labels = self.reshape_ip(labels)
            preds = self.reshape_ip(preds)

        grad = self.loss_derivative(labels, preds).astype(np.float32)

        if restore_dims is True:
            grad.shape = old_shape

        return grad


class BinaryCrossEntropy(Loss):
    name = ["binary_crossentropy", "bce"]
    epsilon = 1e-15
    ndim = 2

    @staticmethod
    def reshape_ip(ip: np.ndarray) -> np.ndarray:
        return ip.reshape(1, -1)

    def loss_func(self, labels: np.ndarray, preds: np.ndarray) -> float:
        if 1.0 in preds or (preds <= 0).any():
            preds = np.clip(preds, self.epsilon, 1.0 - self.epsilon)

        loss = labels * np.log(preds)
        loss += (1 - labels) * np.log(1 - preds)
        loss = np.sum(-loss)

        loss = np.squeeze(loss) / labels.shape[-1]

        return loss

    def loss_derivative(self, labels: np.ndarray, preds: np.ndarray) -> np.ndarray:
        if 1.0 in preds or (preds <= 0).any():
            preds = np.clip(preds, self.epsilon, 1.0 - self.epsilon)

        grad = (1 - labels) / (1 - preds)
        grad -= labels / preds
        grad /= labels.shape[-1]
        return grad


class MeanSquaredError(Loss):
    name = ["mean_squared_error", "mse"]
    ndim = 2

    @staticmethod
    def reshape_ip(ip: np.ndarray) -> np.ndarray:
        return ip.reshape(-1, 1)

    def loss_func(self, labels: np.ndarray, preds: np.ndarray) -> float:
        loss = preds - labels
        loss **= 2
        loss = np.sum(loss / (2 * labels.shape[-1]))

        return np.squeeze(loss)

    def loss_derivative(self, labels: np.ndarray, preds: np.ndarray) -> np.ndarray:
        return (preds - labels) / labels.shape[-1]


class CategoricalCrossEntropy(Loss):
    name = ["categorial_crossentropy", "cce"]
    ndim = 2

    @staticmethod
    def reshape_ip(ip: np.ndarray) -> np.ndarray:
        return ip.reshape(ip.shape[0], -1)

    def loss_func(self, labels: np.ndarray, preds: np.ndarray) -> float:
        loss = -np.sum(labels * np.log(preds), axis=0, keepdims=True)
        loss = np.sum(loss) / labels.shape[-1]
        return np.squeeze(loss)

    def loss_derivative(self, labels: np.ndarray, preds: np.ndarray) -> np.ndarray:
        grad = -labels
        grad /= preds
        grad /= labels.shape[-1]
        return grad
