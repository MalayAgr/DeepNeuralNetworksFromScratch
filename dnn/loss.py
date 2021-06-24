from abc import ABC, abstractmethod

import numpy as np


class Loss(ABC):
    name = None

    def __init__(self, Y, *args, **kwargs):
        self.labels = Y
        self.train_size = Y.shape[-1]

    @staticmethod
    def _add_sub_cls(sub_cls):
        name = sub_cls.name
        if isinstance(name, (list, tuple)):
            return {n: sub_cls for n in name}
        return {name: sub_cls}

    @classmethod
    def get_loss_classes(cls):
        result = {}

        for sub_cls in cls.__subclasses__():
            result.update(sub_cls.get_loss_classes())
            if sub_cls.name is not None:
                result.update(cls._add_sub_cls(sub_cls))
        return result

    def validate_input(self, preds):
        if self.labels.shape != preds.shape:
            raise AttributeError(
                "The labels and the predictions should have the same shape"
            )

    @abstractmethod
    def loss_func(self, preds):
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
    def loss_derivative(self, preds):
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

    def compute_loss(self, preds):
        self.validate_input(preds)
        return self.loss_func(preds)

    def compute_derivatives(self, preds):
        self.validate_input(preds)
        return self.loss_derivative(preds)


class BinaryCrossEntropy(Loss):
    name = ["binary_crossentropy", "bse"]

    def loss_func(self, preds):
        positive_labels = self.labels * np.log(preds)
        negative_labels = (1 - self.labels) * np.log(1 - preds)
        loss = np.sum(-(positive_labels + negative_labels)) / self.train_size
        return np.squeeze(loss)

    def loss_derivative(self, preds):
        lhs = (1 - self.labels) / (1 - preds)
        rhs = self.labels / preds
        return lhs - rhs


class MeanSquaredError(Loss):
    name = ["mean_squared_error", "mse"]

    def loss_func(self, preds):
        loss = np.sum((preds - self.labels) ** 2) / (2 * self.train_size)
        return np.squeeze(loss)

    def loss_derivative(self, preds):
        return preds - self.labels


class CategoricalCrossEntropy(Loss):
    name = ["categorial_crossentropy", "cce"]

    def loss_func(self, preds):
        loss = -np.sum(self.labels * np.log(preds), axis=0, keepdims=True)
        loss = np.sum(loss)
        return np.squeeze(loss)

    def loss_derivative(self, preds):
        return -self.labels / preds
