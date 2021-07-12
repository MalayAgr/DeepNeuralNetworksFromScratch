import numpy as np
from dnn.layers.base_layer import BaseLayer


class Flatten(BaseLayer):
    reset = ("flat",)

    def __init__(self, ip):
        super().__init__(ip=ip, trainable=False)

        self._ip_dims = self.input_shape()[:-1]

        self._units = np.prod(self._ip_dims)

        self.flat = None

    def fans(self):
        _, ip_fan_out = self.ip_layer.fans()

        return ip_fan_out, self.units

    def output(self):
        return self.flat

    def output_shape(self):
        if self.flat is not None:
            return self.flat.shape

        return self._units, None

    def forward_step(self, *args, **kwargs):
        self.flat = self.input().reshape(self._units, -1)

        return self.flat

    def backprop_step(self, dA, *args, **kwargs):
        dA = dA.reshape(*self._ip_dims, -1)

        self.reset_attrs()

        return dA
