"""Contains the abstract base classes for all layer implementations.

Any layer implemented should inherit from one of the classes in this module.

Classes
---------
- BaseLayer: ABC for layers which expect a single Numpy array as input.
- MultiInputBaseLayer: ABC for layers which expect more than one Numpy array as input.
"""


from __future__ import annotations

from abc import ABC, abstractmethod
from itertools import count
from typing import Any, Union

import numpy as np
from dnn.input_layer import Input


class BaseLayer(ABC):
    """Abstract base class for all layers with a single input.

    Attributes
    ----------
    ip_layer: Instance of Input or of a subclass of BaseLayer, or None
        Input to the layer.

    name: str
        Name of the layer (used in computation graph). The name
        should be unique with respect to all other layers in the same
        model.

    trainable: bool
        Indicates whether or not the layer is trainable, i.e. it has parameters
        that are updated during training.

    training: bool
        Indicates whether or not the layer is currently being trained.

    param_keys: list of str
        Names of instance attributes that refer to the parameters of the layer.

    requires_ip_gradient: bool
        Indicates whether or not the layer should compute the gradient wrt its input(s).

    gradients: dict
        Gradients of the layer's parameters (if any).

    built: bool
        Indicates whether the layer has been built or not. When True, it generally
        implies that the parameters of the layer have been initialized.

    reset: tuple
        Attributes that should be reset after every backpropagation step.
        Helps in saving memory between training steps.

    str_attrs: tuple
        Attributes that should make up the str and repr representation of the layer.

    Methods
    ----------
    fans() -> 2-tuple of ints
        Returns the logical number of input and output units of the layer.

    build() -> Any
        Builds the layer, usually by initializing the parameters.

    count_params() -> int
        Counts the number of trainable parameters in the layer.

    input() -> np.ndarray
        Returns the underlying NumPy array which is the input to the layer.

    input_shape() -> tuple
        Returns the expected shape of the input of the layer.

    output() -> nd.array or None:
        Returns the output of the layer.

    output_shape() -> tuple:
        Returns the expected shape of the output of the layer.

    forward_step(*args, **kwargs) -> np.ndarray
        Performs one step of forward propagation.

    transform_backprop_gradient(grad, *args, **kwargs) -> np.ndarray
        Applies transformations on the backpropagated gradient so that it is
        appropriate for use by the layer.
        By default, returns the gradient as is.

    backprop_parameters(grad, *args, **kwargs) -> None
        Computes the gradient of loss wrt the layer's parameters.

    backprop_inputs(grad, *args, **kwargs) -> np.ndarray
        Returns the gradient of loss wrt the layer's input.

    Interface
    ----------
    Subclasses must implement the following methods:
        - forward_step(*args, **kwargs)
        - backprop_inputs(grad, *args, **kwargs)
        - output()
        - output_shape()

    Additionally, subclasses which have the trainable attribute set as True must
    implement these methods:
        - fans()
        - build()
        - count_params()
        - backprop_parameters()
    """

    reset = ()
    str_attrs = ()

    _id = count(0)

    __slots__ = ("__dict__", "param_keys", "gradients")

    def __init__(
        self,
        ip: Input | BaseLayer | None,
        *args,
        trainable: bool = True,
        params: list = None,
        name: str = None,
        **kwargs,
    ) -> None:
        """
        Arguments
        ----------
        ip: Input to the layer.

        trainable: Indicates whether or not the layer is trainable,
        i.e. it has parameters that are updated during training. Defaults to True.

        params: Names of instance attributes that refer to the parameters of the layer.
        Defaults to None.

        name: Name of the layer (used in computation graphs). The name should be unique
        wrt all other layers in the same model. When None, a name is automatically created.
        Defaults to None.

        Raises
        ----------
        TypeError: When ip is not None and it is not an instance of Input or a subclass of BaseLayer.
        """
        if ip is not None and not isinstance(ip, (Input, BaseLayer)):
            msg = (
                f"A {self.__class__.__name__} can have only instances "
                "of Input or a subclass of BaseLayer as ip"
            )
            raise TypeError(msg)

        self.ip_layer = ip

        self.name = self._make_name() if name is None else name

        self.trainable = trainable

        self.training = False

        self._built = False

        self.param_keys = params

        self.requires_ip_gradient = True
        self.gradients = {}

    def __str__(self) -> str:
        attrs = ", ".join(f"{attr}={getattr(self, attr)}" for attr in self.str_attrs)
        return f"{self.__class__.__name__}({attrs})"

    def __repr__(self) -> str:
        return self.__str__()

    @property
    def built(self) -> bool:
        """
        Boolean value that indicates whether or not the layer has been built.
        This usually means that the layer's parameters have been initialized.
        """
        return self._built

    @built.setter
    def built(self, value: bool) -> None:
        self._built = value

    def _make_name(self) -> str:
        """Method to generate a name for the layer."""
        cls = self.__class__.__name__.lower()
        _id = next(BaseLayer._id)  # skipcq: PTC-W0063
        base = f"{cls}{_id}_"

        return base + "/".join(
            f"{attr[0]}@{getattr(self, attr)}"
            for attr in self.str_attrs
            if attr != "activation"
        )

    def fans(self) -> tuple[int, int]:
        """Method to obtain the number of input and output units of the layer.

        Trainable layers must implement this method.

        Raises
        ----------
        NotImplementedError: When the layer is trainable and the method has not
        been implemented.
        """
        if self.trainable:
            raise NotImplementedError(
                f"{self.__class__.__name__} instances need to implement fans."
            )

    def _initializer_variance(self, initializer: str) -> float:
        """Method to obtain the variance for the given initializer.

        Arguments
        ----------
        initializer: Initializer whose variance is required.
        Can be one of "he", "xavier" or "xavier_uniform".

        Raises
        ----------
        KeyError: When initializer is not one of the expected values.
        """
        fan_in, fan_out = self.fans()

        return {
            "he": 2 / fan_in,
            "xavier": 1 / fan_in,
            "xavier_uniform": 6 / (fan_in + fan_out),
        }[initializer]

    def _add_param(self, shape: tuple[int, ...], initializer: str) -> np.ndarray:
        """Helper method to initialize a parameter with the given shape and initializer.

        Arguments:
        ----------
        shape: Shape of the parameter.

        initializer: Initializer that should be used. Supported initializers:
            - "he" - He initializer.
            - "xavier" - Xavier initializer.
            - "xavier_uniform" - Uniform Xavier initializer.
            - "zeros" - Initialize with zeros.
            - "ones" - Initialize with ones.

        Raises
        ----------
        KeyError: When initializer is not one of the above values.
        """
        if initializer == "zeros":
            return np.zeros(shape=shape, dtype=np.float32)
        if initializer == "ones":
            return np.ones(shape=shape, dtype=np.float32)

        variance = self._initializer_variance(initializer)

        return np.random.randn(*shape).astype(np.float32) * np.sqrt(variance)

    def build(self) -> Any:
        """Method to build the layer, usually by initializing the parameters.

        Trainable layers must implement this method.

        Raises
        ----------
        NotImplementedError: When the layer is trainable and the method has not
        been implemented.
        """
        if self.trainable:
            raise NotImplementedError(
                f"{self.__class__.__name__} instances need to implement build."
            )

    def count_params(self) -> int:
        """Method to count the number of trainable parameters in the layer.

        Trainable layers must implement this method.

        Raises
        ----------
        NotImplementedError: When the layer is trainable and the method has not
        been implemented.
        """
        if self.trainable:
            raise NotImplementedError(
                f"{self.__class__.__name__} instances need to implement count_params."
            )
        return 0

    def input(self) -> np.ndarray:
        """Method to obtain the underlying NumPy array which is the input to the layer.

        Raises
        ----------
        AttributeError: When no NumPy array is found to return.
        """
        if self.ip_layer is None:
            raise AttributeError("No input found.")

        ret_val = (
            self.ip_layer.ip
            if isinstance(self.ip_layer, Input)
            else self.ip_layer.output()
        )

        if ret_val is None:
            raise AttributeError("No input found.")

        return ret_val

    def input_shape(self) -> tuple[int, ...]:
        """Method to obtain the expected shape of the input of the layer.

        Raises
        ----------
        AttributeError: When ip_layer is None.
        """
        if self.ip_layer is None:
            raise AttributeError("No input found.")

        if isinstance(self.ip_layer, Input):
            return self.ip_layer.shape

        return self.ip_layer.output_shape()

    @abstractmethod
    def output(self) -> np.ndarray | None:
        """
        Method to obtain the output of the layer.
        """

    @abstractmethod
    def output_shape(self) -> tuple[int, ...]:
        """
        Method to obtain the expected shape of the output of the layer.
        """

    def forward(self, *args, **kwargs) -> np.ndarray:
        """Wrapper around forward_step() which builds the layer if it is not built."""
        if not self.built:
            self.build()
            self.built = True
        return self.forward_step(*args, **kwargs)

    @abstractmethod
    def forward_step(self, *args, **kwargs) -> np.ndarray:
        """
        Method to carry out one step of forward propagation.
        """

    def backprop(self, grad: np.ndarray, *args, **kwargs) -> np.ndarray | None:
        """Method to carry out one complete backprop step.

        Arguments
        ----------
        grad: Backpropagated gradient.
        """
        grad = self.transform_backprop_gradient(grad, *args, **kwargs)

        if self.trainable:
            self.backprop_parameters(grad, *args, **kwargs)

        ret_val = (
            self.backprop_inputs(grad, *args, **kwargs)
            if self.requires_ip_gradient
            else None
        )

        # Clear memory by resetting unnecessary attributes.
        self._reset_attrs()

        return ret_val

    def transform_backprop_gradient(
        self, grad: np.ndarray, *args, **kwargs
    ) -> np.ndarray:
        """Method to apply transformations on the backpropagated gradient so that
        it is appropriate for use by the layer.

        By default, it returns the gradient as is. Override the method
        to apply other transformations like reshape, transpose, activations, etc.

        Arguments
        ----------
        grad: Gradient to be transformed.
        """
        return grad

    def backprop_parameters(self, grad: np.ndarray, *args, **kwargs) -> None:
        """Method to compute the gradient of loss wrt the layer's parameters.

        Trainable layers must implement this method.
        The implementation should store the gradients in the 'gradients'
        attribute of the layer. The gradients should be Numpy arrays and
        the keys should be the same as the values passed as params argument to __init__().

        Arguments
        ----------
        grad: Backpropagated gradient *after* transformation using transform_backprop_gradient().

        Raises
        ----------
        NotImplementedError: When the layer is trainable and the method has not
        been implemented.

        Example
        ---------
        If params = ['weights', 'biases'], then, after this method is called:
            >>> gradients = {
                'weights': <np.ndarray>,
                'biases': <np.ndarray>,
            }
        """
        if self.trainable:
            raise NotImplementedError(
                "Trainable layers must implement backprop_parameters"
            )

    @abstractmethod
    def backprop_inputs(self, grad: np.ndarray, *args, **kwargs) -> np.ndarray:
        """Method to compute the derivative of loss wrt the layer's input.

        It should return a single NumPy array.

        Arguments
        ----------
        grad: Backpropagated gradient *after* transformation using transform_backprop_gradient().
        """

    def _reset_attrs(self) -> None:
        """Method to clean up unnecessary attributes to save memory.

        The method is always called after one complete backprop step.
        """
        for attr in self.reset:
            if isinstance(attr, tuple):
                setattr(self, attr[0], attr[-1])
                continue
            setattr(self, attr, None)


LayerInputType = Union[Input, BaseLayer]


class MultiInputBaseLayer(BaseLayer):
    """Abstract base class for all layers with multiple inputs.

    It inherits from BaseLayer. The class is identical to BaseLayer,
    except for the overrides below.

    Attributes
    ----------
    ip_layer: List of instances of Input or of subclasses of BaseLayer
        Input to the layer.

    Methods
    ----------
    input() -> List of np.ndarray's
        Returns the underlying NumPy arrays which are the input to the layer.

    input_shape() -> List of tuples
        Returns the expected shapes of the inputs of the layer.

    backprop_inputs(grad, *args, **kwargs) -> Tuple of np.ndarray's
        Returns the gradient of loss wrt the layer's inputs.
    """

    def __init__(
        self,
        ip: list[LayerInputType],
        *args,
        trainable: bool,
        params: list = None,
        name: str = None,
        **kwargs,
    ) -> None:
        """
        Arguments
        ----------
        ip: Input to the layer.

        trainable: Indicates whether or not the layer is trainable,
        i.e. it has parameters that are updated during training. Defaults to True.

        params: Names of instance attributes that refer to the
        parameters of the layer. Defaults to None.

        name: Name of the layer (used in computation graphs). The name should be unique
        wrt all other layers in the same model. When None, a name is automatically created.
        Defaults to None.

        Raises
        ----------
        TypeError: When ip is not a list, or when any item in ip
        is not an instance of Input or of subclasses of BaseLayer.
        """
        if ip is not None:
            self._validate_ip(ip)

        super().__init__(
            ip=None, *args, trainable=trainable, params=params, name=name, **kwargs
        )

        self.ip_layer = ip

    def _validate_ip(self, ip):
        """Method to validate the input."""
        if not isinstance(ip, list):
            raise TypeError(f"{self.__class__.__name__} expects a list of inputs.")

        if any(not isinstance(i, (BaseLayer, Input)) for i in ip):
            msg = (
                f"{self.__class__.__name__} expects all inputs in the list of inputs "
                "to be instances of Input or other layers."
            )
            raise TypeError(msg)

    def input(self) -> list[np.ndarray]:
        """Method to obtain the underlying NumPy arrays which are the inputs to the layer.

        Raises
        ----------
        AttributeError: When no Numpy array is found in any of the inputs.
        """
        ret_val = []

        for ip in self.ip_layer:
            op = ip.ip if isinstance(ip, Input) else ip.output()

            if op is None:
                raise AttributeError(f"No input found in {ip}.")

            ret_val.append(op)

        return ret_val

    def input_shape(self) -> list[tuple[int, ...]]:
        """Method to obtain the expected shapes of the inputs of the layer."""
        return [
            ip.shape if isinstance(ip, Input) else ip.output_shape()
            for ip in self.ip_layer
        ]

    def backprop(self, grad: np.ndarray, *args, **kwargs) -> tuple[np.ndarray] | None:
        return super().backprop(grad=grad, *args, **kwargs)

    @abstractmethod
    def backprop_inputs(self, grad: np.ndarray, *args, **kwargs) -> tuple[np.ndarray]:
        """Method to compute the derivative of loss wrt the layer's inputs.

        It should return a tuple of as many NumPy arrays as there are inputs in the layer.

        Arguments
        ----------
        grad: Backpropagated gradient *after* transformation using transform_backprop_gradient().
        """


BaseLayerType = Union[BaseLayer, MultiInputBaseLayer]
