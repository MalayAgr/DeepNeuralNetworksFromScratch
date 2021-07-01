from abc import ABC, abstractmethod

import numpy as np


class Loss(ABC):
    name = None

    @classmethod
    def get_loss_classes(cls):
        result = {}

        for sub_cls in cls.__subclasses__():
            result.update(sub_cls.get_loss_classes())
            if sub_cls.name is not None:
                result.update({name: sub_cls for name in sub_cls.name})
        return result

    def validate_input(self, labels, preds):
        if labels.shape != preds.shape:
            raise AttributeError(
                "The labels and the predictions should have the same shape"
            )

    @abstractmethod
    def loss_func(self, labels, preds):
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
    def loss_derivative(self, labels, preds):
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

    def compute_loss(self, labels, preds):
        self.validate_input(labels, preds)
        return self.loss_func(labels, preds)

    def compute_derivatives(self, labels, preds):
        self.validate_input(labels, preds)
        return self.loss_derivative(labels, preds)


class BinaryCrossEntropy(Loss):
    name = ["binary_crossentropy", "bce"]

    epsilon = 1e-15

    def loss_func(self, labels, preds):
        if 1.0 in preds:
            preds = np.clip(preds, self.epsilon, 1.0 - self.epsilon)

        positive_labels = labels * np.log(preds)
        negative_labels = (1 - labels) * np.log(1 - preds)

        loss = np.sum(-(positive_labels + negative_labels)) / labels.shape[-1]
        return np.squeeze(loss)

    def loss_derivative(self, labels, preds):
        if 1.0 in preds:
            preds = np.clip(preds, self.epsilon, 1.0 - self.epsilon)

        lhs = (1 - labels) / (1 - preds)
        rhs = labels / preds
        return lhs - rhs


class MeanSquaredError(Loss):
    name = ["mean_squared_error", "mse"]

    def loss_func(self, labels, preds):
        loss = np.sum((preds - labels) ** 2) / (2 * labels.shape[-1])
        return np.squeeze(loss)

    def loss_derivative(self, labels, preds):
        return preds - labels


class CategoricalCrossEntropy(Loss):
    name = ["categorial_crossentropy", "cce"]

    def loss_func(self, labels, preds):
        loss = -np.sum(labels * np.log(preds), axis=0, keepdims=True)
        loss = np.sum(loss) / labels.shape[-1]
        return np.squeeze(loss)

    def loss_derivative(self, labels, preds):
        return -labels / preds
