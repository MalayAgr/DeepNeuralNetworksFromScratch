from abc import ABC, abstractmethod

import numpy as np

from dnn.activations import Softmax
from dnn.layer import Layer

from .utils import generate_batches, loss_factory, rgetattr, rsetattr


class Optimizer(ABC):
    def __init__(self, *args, learning_rate=0.01, **kwargs):
        self.lr = learning_rate
        self.train_size = None

    def backprop(self, model, loss, preds):
        dA = loss.compute_derivatives(preds)

        for layer in reversed(model.layers):
            if not hasattr(layer, "param_map"):
                raise AttributeError("No param_map found.")
            layer.backprop_step(dA)
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
