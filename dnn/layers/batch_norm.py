from functools import cached_property
from dnn.layers.base_layer import BaseLayer
import numpy as np


class BatchNorm(BaseLayer):
    reset = ("std", "norm", "scaled_norm")

    def __init__(self, ip, axis=0, momentum=0.5, epsilon=1e-7):
        self.axis = axis
        self.momentum = momentum
        self.epsilon = epsilon

        params = ["gamma", "beta"]

        super().__init__(ip=ip, params=params)

        ip_shape = self.input_shape()
        self.x_dim = ip_shape[axis]
        self.axes = tuple(ax for ax, _ in enumerate(ip_shape) if ax != axis)

        self.std = None
        self.norm = None
        self.scaled_norm = None

        self.mean_mva = None
        self.std_mva = None

    @cached_property
    def fans(self):
        _, ip_fan_out = self.ip_layer.fans

        return ip_fan_out, ip_fan_out

    def init_params(self):
        rem_dims = (1,) * len(self.axes)

        self.gamma = np.ones(shape=(self.x_dim, *rem_dims), dtype=np.float32)

        self.beta = np.zeros(shape=(self.x_dim, *rem_dims), dtype=np.float32)

    def _init_mva(self):
        rem_dims = (1,) * len(self.axes)

        self.mean_mva = np.zeros(shape=(self.x_dim, *rem_dims), dtype=np.float32)

        self.std_mva = np.ones(shape=(self.x_dim, *rem_dims), dtype=np.float32)

    def count_params(self):
        return 2 * self.x_dim

    def build(self):
        self.init_params()
        self._init_mva()

    def output(self):
        return self.scaled_norm

    def output_shape(self):
        return self.input_shape()

    def _update_mva(self, mean, std):
        mom, one_minus_mom = self.momentum, 1 - self.momentum

        self.mean_mva *= mom
        self.mean_mva += one_minus_mom * mean

        self.std_mva *= mom
        self.std_mva += one_minus_mom * mean

    def forward_step(self, *args, **kwargs):
        ip = self.input()

        if self.is_training:
            mean = ip.mean(axis=self.axes, keepdims=True)
            std = np.sqrt(ip.var(axis=self.axes, keepdims=True) + self.epsilon)

            self._update_mva(mean, std)

            self.std = std
        else:
            mean, std = self.mean_mva, self.std_mva

        self.norm = ip - mean
        self.norm /= std

        self.scaled_norm = self.gamma * self.norm + self.beta

        return self.scaled_norm

    def backprop_step(self, dA, *args, **kwargs):
        self.gradients = {
            "gamma": np.sum(dA * self.norm, axis=self.axes, keepdims=True),
            "beta": np.sum(dA, axis=self.axes, keepdims=True),
        }

        dA *= self.gamma

        mean_share = dA.sum(axis=self.axes, keepdims=True)
        var_share = self.norm * np.sum(dA * self.norm, axis=self.axes, keepdims=True)

        scale = dA.size / self.x_dim

        if self.requires_dX is False:
            self.reset_attrs()
            return

        dX = (scale * dA - mean_share - var_share) / (scale * self.std)

        self.reset_attrs()

        return dX
