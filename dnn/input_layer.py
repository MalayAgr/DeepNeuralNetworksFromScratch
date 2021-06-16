import numpy as np


class Input:
    def __init__(self, shape, *args, **kwargs):
        self.ip_shape = shape
        self._ip = None

    @property
    def ip(self):
        return self._ip

    @ip.setter
    def ip(self, X):
        if X.shape != self.ip_shape:
            raise AttributeError("The input does not have the expected shape")
        self._ip = X
