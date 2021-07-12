from functools import cached_property

import numpy as np
from dnn.layers.base_layer import BaseLayer


class Dropout(BaseLayer):
    reset = ("dropped", "dropout_mask")

    def __init__(self, ip, keep_prob=0.5):
        if not 0 < keep_prob <= 1:
            raise AttributeError("keep_prob should be in the interval (0, 1]")

        self.keep_prob = keep_prob

        super().__init__(ip=ip, trainable=False)

        self.dropped = None
        self.dropout_mask = None

    @cached_property
    def fans(self):
        _, ip_fan_out = self.ip_layer.fans

        return ip_fan_out, ip_fan_out

    def output(self):
        return self.dropped

    def output_shape(self):
        return self.input_shape()

    def forward_step(self, *args, **kwargs):
        ip = self.input()

        self.dropout_mask = (
            np.random.rand(*ip.shape).astype(np.float32) < self.keep_prob
        )

        self.dropped = (ip * self.dropout_mask) / self.keep_prob

        return self.dropped

    def backprop_step(self, dA, *args, **kwargs):
        dA *= self.dropout_mask
        dA /= self.keep_prob

        self.reset_attrs()

        return dA
