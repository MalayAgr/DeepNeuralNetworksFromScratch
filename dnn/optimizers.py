from abc import ABC, abstractmethod

import numpy as np

from dnn.activations import Softmax
from dnn.layer import Layer

from .utils import generate_batches, loss_factory, rgetattr, rsetattr


class Optimizer(ABC):
    def __init__(self, *args, learning_rate=0.01, **kwargs):
        self.lr = learning_rate
        self.train_size = None

    @staticmethod
    def get_layer_dA(dA_params):
        if isinstance(dA_params, Layer):
            next_weights = dA_params.weights
            next_dZ = dA_params.dZ

            return np.matmul(next_weights.T, next_dZ)
        return dA_params

    def get_layer_dZ(self, layer, dA_params):
        layer_dA = self.get_layer_dA(dA_params)

        if layer.batch_norm is not False:
            bn = layer.batch_norm
            activation_grads = layer.activation.calculate_derivatives(bn.norm)

            d_norm = (
                np.sum(layer_dA * activation_grads, axis=1)
                if isinstance(layer.activation, Softmax)
                else layer_dA * activation_grads
            )

            grads = {
                "gamma": np.sum(d_norm * bn.Z_hat, axis=1, keepdims=True),
                "beta": np.sum(d_norm, axis=1, keepdims=True),
            }

            dZ_hat = d_norm * bn.gamma

            dZ_hat_sum = np.sum(dZ_hat, axis=1, keepdims=True)
            dZ_hat_prod = bn.Z_hat * np.sum(dZ_hat * bn.Z_hat, axis=1, keepdims=True)

            bs = layer.get_ip().shape[-1]

            return (bs * dZ_hat - dZ_hat_sum - dZ_hat_prod) / (bs * bn.std), grads

        activation_grads = layer.activation.calculate_derivatives(layer.linear)

        dZ = (
            np.sum(layer_dA * activation_grads, axis=1)
            if isinstance(layer.activation, Softmax)
            else layer_dA * activation_grads
        )

        return dZ, {}

    def layer_gradients(self, layer, dA_params):
        dZ, grads = self.get_layer_dZ(layer, dA_params)

        gradients = {
            "weights": np.matmul(dZ, layer.get_ip().T) / self.train_size,
            "biases": np.sum(dZ, keepdims=True, axis=1) / self.train_size,
            **grads,
        }

        layer.dZ = dZ
        layer.gradients = gradients

        return gradients

    def backprop(self, model, loss, preds):
        dA = loss.compute_derivatives(preds)

        for layer in reversed(model.layers):
            if not hasattr(layer, "param_map"):
                raise AttributeError("No param_map found.")
            self.layer_gradients(layer, dA)
            dA = layer

    @abstractmethod
    def optimize(self, model, X, Y, batch_size, epochs, loss="bse", shuffle=True):
        pass


class SGD(Optimizer):
    def __init__(self, *args, learning_rate=0.01, **kwargs):
        momentum = kwargs.pop("momentum", 0)

        if not 0 <= momentum <= 1:
            raise AttributeError("momentum should be between 0 and 1")

        self.momentum = momentum
        super().__init__(*args, learning_rate=learning_rate, **kwargs)

    @staticmethod
    def init_velocities(model):
        for layer in model.layers:
            layer.velocities = {
                key: np.zeros(shape=rgetattr(layer, attr).shape)
                for key, attr in layer.param_map.items()
            }

    def update_layer_velocities(self, layer):
        for key, grad in layer.gradients.items():
            lhs = self.momentum * layer.velocities[key]
            rhs = (1 - self.momentum) * grad
            layer.velocities[key] = lhs + rhs

    def update_params_momentum(self, model):
        for layer in model.layers:
            self.update_layer_velocities(layer)
            for key, attr in layer.param_map.items():
                current_val = rgetattr(layer, attr)
                rsetattr(layer, attr, current_val - self.lr * layer.velocities[key])

    def update_params_no_momentum(self, model):
        for layer in model.layers:
            for key, attr in layer.param_map.items():
                current_val = rgetattr(layer, attr)
                rsetattr(layer, attr, current_val - self.lr * layer.gradients[key])

    def get_update_function(self, model):
        if self.momentum == 0:
            return self.update_params_no_momentum

        self.init_velocities(model)
        return self.update_params_momentum

    def optimize(self, model, X, Y, batch_size, epochs, loss="bse", shuffle=True):
        cost, history = 0, []

        update_func = self.get_update_function(model)

        for epoch in range(epochs):
            batches = generate_batches(X, Y, batch_size=batch_size, shuffle=shuffle)
            for batch_X, batch_Y, size in batches:
                self.train_size = size
                # Forward pass
                preds = model.predict(batch_X)
                # Compute cost
                loss_func = loss_factory(loss, batch_Y)
                cost = loss_func.compute_loss(preds)
                # Backprop
                self.backprop(model, loss=loss_func, preds=preds)
                # Update params
                update_func(model)

            print(f"Loss at the end of epoch {epoch + 1}: {cost: .9f}")
            history.append(cost)

        return history


class RMSProp(Optimizer):
    def __init__(self, *args, learning_rate=0.01, **kwargs):
        rho = kwargs.pop("rho", 0.9)

        if not 0 <= rho <= 1:
            raise AttributeError("rho should be between 0 and 1")

        self.rho = rho
        self.epsilon = kwargs.pop("epislon", 1e-7)
        super().__init__(*args, learning_rate=learning_rate, **kwargs)

    @staticmethod
    def init_rms(model):
        for layer in model.layers:
            layer.rms = {
                key: np.zeros(shape=rgetattr(layer, attr).shape)
                for key, attr in layer.param_map.items()
            }

    def update_layer_rms(self, layer):
        for key, grad in layer.gradients.items():
            lhs = self.rho * layer.rms[key]
            rhs = (1 - self.rho) * np.square(grad)
            layer.rms[key] = lhs + rhs

    def get_update(self, grad, rms):
        return self.lr * (grad / (np.sqrt(rms) + self.epsilon))

    def update_params(self, model):
        for layer in model.layers:
            self.update_layer_rms(layer)
            for key, attr in layer.param_map.items():
                current_val = rgetattr(layer, attr)
                update = self.get_update(layer.gradients[key], layer.rms[key])
                rsetattr(layer, attr, current_val - update)

    def optimize(self, model, X, Y, batch_size, epochs, loss="bse", shuffle=True):
        cost, history = 0, []

        self.init_rms(model)

        for epoch in range(epochs):
            batches = generate_batches(X, Y, batch_size=batch_size, shuffle=shuffle)
            for batch_X, batch_Y, size in batches:
                self.train_size = size
                # Forward pass
                preds = model.predict(batch_X)
                # Compute cost
                loss_func = loss_factory(loss, batch_Y)
                cost = loss_func.compute_loss(preds)
                # Backprop
                self.backprop(model, loss=loss_func, preds=preds)
                # Update params
                self.update_params(model)

            print(f"Loss at the end of epoch {epoch + 1}: {cost: .9f}")
            history.append(cost)

        return history


class Adam(Optimizer):
    def __init__(self, *args, learning_rate=0.01, **kwargs):
        momentum = kwargs.pop("beta_1", 0.9)

        if not 0 <= momentum <= 1:
            raise AttributeError("alpha should be between 0 and 1")

        rho = kwargs.pop("beta_2", 0.999)

        if not 0 <= rho <= 1:
            raise AttributeError("beta should be between 0 and 1")

        self.momentum = momentum
        self.rho = rho
        self.epsilon = kwargs.pop("epsilon", 1e-8)
        super().__init__(*args, learning_rate=learning_rate, **kwargs)

    @staticmethod
    def init_moments(model):
        for layer in model.layers:
            layer.m1 = {
                key: np.zeros(shape=rgetattr(layer, attr).shape)
                for key, attr in layer.param_map.items()
            }
            layer.m2 = {
                key: np.zeros(shape=rgetattr(layer, attr).shape)
                for key, attr in layer.param_map.items()
            }

    def update_layer_m1(self, layer):
        for key, grad in layer.gradients.items():
            lhs = self.momentum * layer.m1[key]
            rhs = (1 - self.momentum) * grad
            layer.m1[key] = lhs + rhs

    def update_layer_m2(self, layer):
        for key, grad in layer.gradients.items():
            lhs = self.rho * layer.m2[key]
            rhs = (1 - self.rho) * np.square(grad)
            layer.m2[key] = lhs + rhs

    def get_update(self, m1, m2, t):
        m1 = np.divide(m1, 1.0 - self.momentum ** t)
        m2 = np.divide(m2, 1.0 - self.rho ** t)

        return self.lr * (m1 / (np.sqrt(m2) + self.epsilon))

    def update_params(self, model, t):
        for layer in model.layers:
            self.update_layer_m1(layer)
            self.update_layer_m2(layer)

            for key, attr in layer.param_map.items():
                current_val = rgetattr(layer, attr)
                update = self.get_update(layer.m1[key], layer.m2[key], t)
                rsetattr(layer, attr, current_val - update)

    def optimize(self, model, X, Y, batch_size, epochs, loss="bse", shuffle=True):
        cost, history = 0, []

        self.init_moments(model)

        t = 0
        for epoch in range(epochs):
            batches = generate_batches(X, Y, batch_size=batch_size, shuffle=shuffle)
            for batch_X, batch_Y, size in batches:
                self.train_size = size
                # Forward pass
                preds = model.predict(batch_X)
                # Compute cost
                loss_func = loss_factory(loss, batch_Y)
                cost = loss_func.compute_loss(preds)
                # Backprop
                self.backprop(model, loss=loss_func, preds=preds)
                # Update params
                t += 1
                self.update_params(model, t)

            print(f"Loss at the end of epoch {epoch + 1}: {cost: .9f}")
            history.append(cost)

        return history
