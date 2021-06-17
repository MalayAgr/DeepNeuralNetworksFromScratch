from abc import ABC, abstractmethod
import numpy as np
from .utils import loss_factory
from dnn.layer import Layer


class Optimizer(ABC):
    def __init__(self, learning_rate, *args, loss="bse", **kwargs):
        self.lr = learning_rate
        self.loss = loss

    def get_layer_dA(self, dA_params):
        if isinstance(dA_params, Layer):
            next_weights = dA_params.weights
            next_dZ = dA_params.dZ

            return np.matmul(next_weights.T, next_dZ)
        return dA_params

    def layer_gradients(self, layer, dA_params):
        layer_dA = self.get_layer_dA(dA_params)

        dZ = layer_dA * layer.activation.calculate_derivatives(layer.linear)

        gradients = {
            "weights": np.matmul(dZ, layer.get_ip().T) / self.train_size,
            "biases": np.sum(dZ, keepdims=True, axis=1) / self.train_size,
        }

        layer.dZ = dZ
        layer.gradients = gradients

        return gradients

    def backprop(self, model, loss, preds):
        dA = loss.compute_derivatives(preds)

        for layer in reversed(model.layers):
            self.layer_gradients(layer, dA)
            dA = layer

    @abstractmethod
    def update_params(self, *args, **kwargs):
        pass

    @abstractmethod
    def optimize(self, model, X, Y, batch_size, epochs, shuffle=True):
        pass
