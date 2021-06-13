import numpy as np
from abc import abstractmethod


class Loss:
    def __init__(self, Y):
        self.labels = Y
        self.train_size = Y.shape[-1]

    def validate_input(self, preds):
        if self.Y.shape != self.preds.shape:
            raise AttributeError(
                "The labels and the predictions should have the same shape"
            )

    @abstractmethod
    def loss_func(self, preds):
        pass

    @abstractmethod
    def loss_derivative(self, preds):
        pass

    def compute_loss(self, preds):
        return self.loss_func(preds)

    def compute_derivatives(self, preds):
        return self.loss_derivative(preds)


class BinaryCrossEntropy(Loss):
    def loss_func(self, preds):
        positive_labels = self.labels * np.log(preds)
        negative_labels = (1 - self.labels) * np.log(1 - preds)
        loss = np.sum((-(positive_labels + negative_labels))) / self.train_size
        return np.squeeze(loss)

    def loss_derivative(self, preds):
        lhs = (1 - self.labels) / (1 - preds)
        rhs = self.labels / preds
        return lhs - rhs


def loss_factory(self, loss, Y):
    return {
        "binary_crossentropy": BinaryCrossEntropy(Y),
        "bse": BinaryCrossEntropy(Y),
    }[loss]
