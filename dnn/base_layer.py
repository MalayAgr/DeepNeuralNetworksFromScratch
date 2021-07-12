from abc import ABC, abstractmethod
from functools import cached_property

from dnn.input_layer import Input


class BaseLayer(ABC):
    reset = None

    def __init__(self, ip, *args, trainable=True, params=None, **kwargs):
        if ip is not None and not isinstance(ip, (Input, BaseLayer)):
            msg = (
                f"A {self.__class__.__name__} can have only instances "
                "of Input or a subclass of BaseLayer as ip"
            )
            raise AttributeError(msg)

        self.ip_layer = ip

        self.trainable = trainable

        self.is_training = False

        if params is not None:
            self.param_map = self._add_params(params)

        self.requires_dX = True
        self.gradients = {}

        self._add_extra_attrs(kwargs)

    def _add_params(self, params):
        for param in params:
            self.__setattr__(param, None)

        return {param: param for param in params}

    def _add_extra_attrs(self, attrs):
        for attr, value in attrs.items():
            self.__setattr__(attr, value)

    @cached_property
    @abstractmethod
    def fans(self):
        """
        Method to obtain the number of input and output units
        """

    def _initializer_variance(self, initializer):
        fan_in, fan_out = self.fans

        return {
            "he": 2 / fan_in,
            "xavier": 1 / fan_in,
            "xavier_uniform": 6 / (fan_in + fan_out),
        }[initializer]

    def count_params(self):
        """
        Method to count the number of trainable parameters in the layer
        """
        if not hasattr(self, "param_map"):
            return 0
        raise NotImplementedError(
            f"{self.__class__.__name__} instances need to implement count_params"
        )

    def build(self):
        """
        Method to build the layer, usually by initializing the parameters
        """
        if hasattr(self, "param_map"):
            raise NotImplementedError(
                f"{self.__class__.__name__} instances need to implement build"
            )

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

    @abstractmethod
    def output(self):
        """
        Method to obtain the output of the layer
        """

    @abstractmethod
    def output_shape(self):
        """
        Method to determine the shape of the output of the layer
        """

    @abstractmethod
    def forward_step(self, *args, **kwargs):
        """
        One step of forward propagation
        """

    @abstractmethod
    def backprop_step(self, dA, *args, **kwargs):
        """
        One step of backpropagation
        """

    def reset_attrs(self):
        for attr in self.reset:
            if isinstance(attr, tuple):
                setattr(self, attr[0], attr[-1])
                continue
            setattr(self, attr, None)
