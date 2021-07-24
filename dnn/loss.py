from __future__ import annotations

from abc import ABC, abstractmethod
from typing import List, Tuple

import numpy as np


class Loss(ABC):
    name: List = None
    ndim: int = None

    @classmethod
    def get_loss_classes(cls) -> dict:
        result = {}

        for sub_cls in cls.__subclasses__():
            result.update(sub_cls.get_loss_classes())
            if sub_cls.name is not None:
                result.update({name: sub_cls for name in sub_cls.name})
        return result

    def validate_input(self, labels: np.ndarray, preds: np.ndarray) -> None:
        if labels.shape != preds.shape:
            raise AttributeError(
                "The labels and the predictions should have the same shape"
            )

        if labels.ndim < self.ndim:
            raise AttributeError(
                f"{self.__class__.__name__} expects at least {self.ndim}-dimensional inputs"
            )

    def should_reshape(self, shape: Tuple) -> bool:
        """Method to determine if the labels and predictions should be reshaped."""
        return False

    @staticmethod
    def reshape_labels_and_preds(
        labels: np.ndarray, preds: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
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

        return self.loss_func(labels, preds).astype(np.float32)

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
    name = ["binary_crossentropy", "bce"]
    ndim = 2
    epsilon = 1e-15

    def should_reshape(self, shape: Tuple) -> bool:
        return len(shape) > self.ndim or shape[0] != 1

    @staticmethod
    def reshape_labels_and_preds(
        labels: np.ndarray, preds: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        return labels.reshape(1, -1), preds.reshape(1, -1)

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

    def should_reshape(self, shape: Tuple) -> bool:
        return len(shape) > self.ndim or shape[0] != 1

    @staticmethod
    def reshape_labels_and_preds(
        labels: np.ndarray, preds: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
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
    name = ["categorial_crossentropy", "cce"]
    ndim = 2

    def should_reshape(self, shape: Tuple) -> bool:
        return len(shape) > self.ndim

    @staticmethod
    def reshape_labels_and_preds(
        labels: np.ndarray, preds: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        classes = labels.shape[0]
        return labels.reshape(classes, -1), preds.reshape(classes, -1)

    def loss_func(self, labels: np.ndarray, preds: np.ndarray) -> float:
        loss = -np.sum(labels * np.log(preds), axis=0, keepdims=True)
        loss = np.sum(loss) / labels.shape[-1]
        return np.squeeze(loss)

    def loss_derivative(self, labels: np.ndarray, preds: np.ndarray) -> np.ndarray:
        grad = -labels
        grad /= preds
        grad /= labels.shape[-1]
        return grad
