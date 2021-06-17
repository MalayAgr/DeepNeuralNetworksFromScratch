import numpy as np

from .input_layer import Input
from .layer import Layer
from .utils import activation_factory, loss_factory


class Model:
    def __init__(self, ip_shape, layer_sizes, activations, initializers=None):
        num_layers = len(layer_sizes)

        if len(activations) != num_layers:
            raise AttributeError(
                "activations and layer_sizes should have the same length"
            )

        if initializers is not None and len(initializers) != num_layers:
            raise AttributeError(
                "initializers and layer_sizes should have the same length"
            )

        self.ip_layer = Input(ip_shape)
        self.layer_sizes = layer_sizes
        self.num_layers = num_layers
        self.activations = activations
        self.initializers = (
            [None] * num_layers if initializers is None else initializers
        )
        self.layers = self._build_model()

    def __str__(self):
        layers = ", ".join([str(l) for l in self.layers])
        return f"{self.__class__.__name__}({self.ip_layer}, {layers})"

    def __repr__(self):
        return self.__str__()

    def _build_model(self):
        ip = self.ip_layer
        inits = self.initializers

        layers = ()
        for size, activation, init in zip(self.layer_sizes, self.activations, inits):
            if init is not None:
                layer = Layer(
                    ip=ip, units=size, activation=activation, initializer=init
                )
            else:
                layer = Layer(ip=ip, units=size, activation=activation)
            layers += (layer,)
            ip = layer

        return layers

    def predict(self, X):
        self.ip_layer.ip = X

        for layer in self.layers:
            layer.forward_step()

        return self.layers[-1].activations

    def train(self, X, Y, batch_size, epochs, lr=1e-3, opt="sgd", loss="bse"):
        pass
