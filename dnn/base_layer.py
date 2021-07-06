from functools import cached_property
from dnn.input_layer import Input


class BaseLayer:
    def __init__(self, ip, *args, trainable=True, params=None, **kwargs):
        if ip is not None:
            if not isinstance(ip, (Input, BaseLayer)):
                msg = f"A {self.__class__.__name__} can have only instances of Input or other layers as ip"
                raise AttributeError(msg)

        self.ip_layer = ip

        self.trainable = trainable

        self.is_training = False

        if params is not None:
            self.param_map = self._add_params(params)

        self._dX = None
        self.gradients = {}

        self._add_extra_attrs(kwargs)

    def _add_params(self, params):
        for param in params:
            self.__setattr__(param, None)

        return {param: param for param in params}

    def _add_extra_attrs(self, attrs):
        for attr, value in attrs.items():
            self.__setattr__(attr, value)

    @property
    def dX(self):
        return self._dX

    @dX.setter
    def dX(self, value):
        self._dX = value

    @cached_property
    def fans(self):
        ...

    def _initializer_variance(self, initializer):
        fan_in, fan_out = self.fans

        return {
            "he": 2 / fan_in,
            "xavier": 1 / fan_in,
            "xavier_uniform": 6 / (fan_in + fan_out),
        }[initializer]

    def init_params(self):
        ...

    def count_params(self):
        ...

    def build(self):
        ...

    def input(self):
        if self.ip_layer is None:
            raise ValueError("No input found")

        ret_val = (
            self.ip_layer.ip
            if isinstance(self.ip_layer, Input)
            else self.ip_layer.output()
        )

        if ret_val is None:
            raise ValueError("No input found")

        return ret_val

    def input_shape(self):
        if isinstance(self.ip_layer, Input):
            return self.ip_layer.ip_shape
        return self.ip_layer.output_shape()

    def output(self):
        ...

    def output_shape(self):
        ...

    def forward_step(self, *args, **kwargs):
        ...

    def backprop_step(self, dA, *args, **kwargs):
        ...
