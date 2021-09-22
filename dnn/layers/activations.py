from __future__ import annotations

from abc import abstractmethod
from typing import Dict, Optional

import numpy as np

from .base_layer import BaseLayer, LayerInput


class Activation(BaseLayer):
    """Abstract base class for all activations.

    It behaves like a layer but can also be supplied input externally.

    Inherits From
    ----------
    BaseLayer

    Attributes
    ----------
    activations: np.ndarray
        Calculated activations for the given input.

    name: str
        Class attribute used to identify an activation class uniquely
        and to discover user-defined activations automatically. Note:
        It is different from the name attribute inherited from BaseLayer.

    Methods
    ---------
    get_activation_classes() -> dict
        Class method which returns a dictionary containing the class attribute name
        as the key and an instance of Activation as the value. Essentially acts as
        a registry for activations.

    activation_func(ip: np.ndarray) -> np.ndarray
        Returns the computed activations for the given input.

    derivative_func(ip: np.ndarray, activations: Optional[np.ndarray] = None) -> np.ndarray
        Returns the derivative of the activation function with respect to the given input.

    compute_activations(ip: Optional[np.ndarray] = None) -> np.ndarray
        Returns the computed activations for either the given input or the passed
        input when the class is used as a layer. In case both are available,
        the given input takes precedence.

    compute_derivatives(ip: Optional[np.ndarray] = None) -> np.ndarray
        Returns the derivative of the activation function with respect to either the
        given input or the passed input when used as a layer. In case both are available,
        the given input takes precedence.

    Interface
    ----------
    Subclasses must implement the following methods:
        - activation_func(ip)
        - derivative_func(ip, activations)
    """

    name = None
    reset = ("activations",)

    def __init__(
        self, *args, ip: Optional[LayerInput] = None, name: str = None, **kwargs
    ) -> None:
        """
        Arguments
        ---------
        ip: Input to the layer. Defaults to None, which is useful when the activation
        is not part of some model.

        name: Name of the layer (used in computation graphs). The name should be unique
        wrt all other layers in the same model. When None, a name is automatically created.
        Defaults to None.
        """
        super().__init__(ip=ip, trainable=False, name=name)

        self.activations = None

    @classmethod
    def get_activation_classes(cls) -> Dict[str, Activation]:
        """Class method to obtain a registry of all subclasses of Activation.

        This allows users to refer to activations by a string name
        instead of importing a class. The registry is a dictionary.
        Each value in the dictionary is a subclass of Activation and
        the key is the corresponding value of the class attribute name
        of the subclass.

        Example
        ----------
        Consider the following user-defined subclass.
        Assume necessary methods have been implemented:

        >>> class UserActivation(Activation):
        ...     name = 'user_activation'

        Then, using the method:

        >>> Activation.get_activation_classes()
        {'linear': dnn.layers.activations.Linear,
        'sigmoid': dnn.layers.activations.Sigmoid,
        ...
        'user_activation': __main__.UserActivation}

        To initialize:

        >>> registry = Activation.get_activation_classes()
        >>> cls = registry['user_activation']
        >>> act = cls()
        >>> type(act)
        __main__.UserActivation
        """
        result = {}

        for sub_cls in cls.__subclasses__():
            name = sub_cls.name
            # Find subclasses of subclasses
            result.update(sub_cls.get_activation_classes())
            if name is not None and name not in result:
                result.update({name: sub_cls})
        return result

    @abstractmethod
    def activation_func(self, ip: np.ndarray) -> np.ndarray:
        """The formula used to calculate the activations.

        If the activation function is called g with input ip,
        this should return g(ip).

        Arguments
        ----------
        ip (Numpy-array): The input for the function.

        Example
        ----------
        The sigmoid activation function is defined as 1 / (1 + e^(-ip)).

        This can be implemented as:
        >>> def activation_func(self, ip: np.ndarray) -> np.ndarray:
        ...     return 1 / (1 + np.exp(-ip))
        """

    @abstractmethod
    def derivative_func(
        self, ip: np.ndarray, activations: Optional[np.ndarray] = None
    ) -> np.ndarray:
        """The formula used to calculate the derivatives.

        If the activation function is called g with input ip,
        this should return g'(ip).

        Arguments
        ----------
        ip: The input with respect to which derivatives
        need to be calculated.

        activations: The activations for the given input.
        This will be passed when the activation is used as a layer, preventing
        recalculating it for activations which require the value.

        Example
        ----------
        For sigmoid, the derivative is sigmoid(ip) * (1 - sigmoid(ip)).

        This can be implemented as:
        >>> def derivative_func(
        ...     self,
        ...     ip: np.ndarray,
        ...     activations: Optional[np.ndarray] = None
        ... ) -> np.ndarray:
        ...     if activations is None:
        ...         activations = self.activation_func(ip)
        ...     return activations * (1 - activations)
        """

    def compute_activations(self, ip: Optional[np.ndarray] = None) -> np.ndarray:
        """Method to handle whether the input is being supplied externally or
        being passed by some layer, and calculate the activations accordingly.

        Arguments
        ---------
        ip: External input. When not None, this will be used to calculate the activations.
        Defaults to None.
        """
        if ip is None:
            ip = self.input()

        return self.activation_func(ip).astype(np.float32)

    def compute_derivatives(self, ip: Optional[np.ndarray] = None) -> np.ndarray:
        """Method to handle whether the input is being supplied externally or
        being passed by some layer, and calculate the derivatives accordingly.

        Arguments
        ---------
        ip: External input. When not None, this will be used to calculate the derivatives.
        Defaults to None.
        """
        activations = None

        if ip is None:
            ip = self.input()
            activations = self.activations

        return self.derivative_func(ip, activations=activations).astype(np.float32)

    def output(self) -> Optional[np.ndarray]:
        return self.activations

    def output_shape(self):
        return self.input_shape()

    def forward_step(self, *args, **kwargs) -> np.ndarray:
        ip = kwargs.pop("ip", None)

        self.activations = self.compute_activations(ip)

        return self.activations

    def backprop_inputs(self, grad: np.ndarray, *args, **kwargs) -> np.ndarray:
        ip = kwargs.pop("ip", None)
        return grad * self.compute_derivatives(ip)


class Linear(Activation):
    """Linear activation (i.e. No activation).

    Definition
    ----------
    linear(ip) = ip

    Derivative
    ----------
    linear'(ip) = 1

    Input shape
    ----------
    (..., batch_size), where ... represents any number of dimensions.

    Output shape
    ----------
    Same as the input shape.

    Example
    ----------
    >>> from dnn.layers.activations import Linear
    >>> import numpy as np
    >>> x = np.array([1, 2, 3, 4, 5])

    >>> linear = Linear()

    >>> linear.compute_activations(x)
    array([1., 2., 3., 4., 5.], dtype=float32)

    >>> linear.compute_derivatives(x)
    array([1., 1., 1., 1., 1.], dtype=float32)
    """

    name = "linear"

    def __str__(self) -> str:
        return "None"

    def activation_func(self, ip: np.ndarray) -> np.ndarray:
        return ip

    def derivative_func(
        self, ip: np.ndarray, activations: Optional[np.ndarray] = None
    ) -> np.ndarray:
        return np.ones_like(ip, dtype=np.float32)


class Sigmoid(Activation):
    """Sigmoid activation.

    Definition
    ----------
    sigmoid(ip) = 1 / (1 + e^(-ip)), where e is Euler's constant.

    Derivative
    ----------
    sigmoid'(ip) = sigmoid(ip) * (1 - sigmoid(ip))

    Input shape
    ----------
    (..., batch_size), where ... represents any number of dimensions.

    Output shape
    ----------
    Same as the input shape.

    Example
    ----------
    >>> from dnn.layers.activations import Sigmoid
    >>> import numpy as np
    >>> x = np.array([-2, -1, 0, 1, 2])

    >>> sigmoid = Sigmoid()

    >>> sigmoid.compute_activations(x)
    array([0.11920292, 0.26894143, 0.5, 0.7310586, 0.8807971 ], dtype=float32)

    >>> sigmoid.compute_derivatives(x)
    array([0.10499358, 0.19661193, 0.25, 0.19661193, 0.10499358], dtype=float32)
    """

    name = "sigmoid"

    def activation_func(self, ip: np.ndarray) -> np.ndarray:
        z = 1.0
        z /= np.exp(-ip) + 1
        return z

    def derivative_func(
        self, ip: np.ndarray, activations: Optional[np.ndarray] = None
    ) -> np.ndarray:
        if activations is None:
            activations = self.activation_func(ip)

        return activations * (1 - activations)


class Softmax(Activation):
    """Softmax activation.

    The softmax activation is unique since it is defined only
    for vectors. It takes a vector as input and outputs another vector.
    Its derivative is a matrix.

    Definition
    ----------
    softmax(ip) = e^(ip) / sum(e^(ip)), where e is Euler's constant.

    Derivative
    ----------
    Consider that ip is a c-dimensional column vector, [i_1, i_2, ..., i_c] and
    softmax(ip) is another c-dimensional column vector, [s_1, s_2, ..., s_c]. Then,
    the derivate is a c x c matrix where the entry in row i and column j is defined as:
        - s_i * (1 - s_i) if i = j
        - -(s_i * s_j) if i != j

    eg - If ip is 3-dimensional, then the derivative is a 3 x 3 matrix:
    >>> [
    ...     [s_1 * (1 - s_1), -s_1 * s_2, -s_1 * s_3],
    ...     [-s_2 * s_1, s_2 * (1 - s_2), -s_2 * s_3],
    ...     [-s_3 * s_1, -s_3 * s_2, s_3 * (1 - s_3)],
    ... ]

    When there is more than one vector, i.e. when ip is a c x n matrix,
    where c is the number of dimensions in each vector and n is the number of vectors,
    then the derivative is c x c x n, with one c x c matrix for each vector.

    Input shape
    ----------
    (c, batch_size), where c is the number of categories.

    Output shape
    ----------
    Same as the input shape.

    Example
    ----------
    >>> from dnn.layers.activations import Softmax
    >>> import numpy as np
    >>> x = np.array([-2, -1, 0, 1, 2]).reshape(-1, 1)  # Make column vector

    >>> softmax = Softmax()

    >>> activations = softmax.compute_activations(x)
    >>> activations.shape
    (5, 1)

    >>> activations
    array([[0.01165623],
           [0.03168492],
           [0.08612855],
           [0.23412165],
           [0.6364086 ]], dtype=float32)

    >>> derivatives = softmax.compute_derivatives(x)
    >>> derivatives.shape
    (5, 5, 1)

    >>> derivatives[..., 0]  # Extract the derivative
    array([[ 0.01152036, -0.00036933, -0.00100393, -0.00272898, -0.00741813],
           [-0.00036933,  0.03068099, -0.00272898, -0.00741813, -0.02016456],
           [-0.00100393, -0.00272898,  0.07871042, -0.02016456, -0.05481295],
           [-0.00272898, -0.00741813, -0.02016456,  0.17930871, -0.14899705],
           [-0.00741813, -0.02016456, -0.05481295, -0.14899705,  0.23139268]],
          dtype=float32)
    """

    name = "softmax"

    def activation_func(self, ip: np.ndarray) -> np.ndarray:
        z = ip - np.max(ip, axis=0, keepdims=True)
        z = np.exp(z)
        z /= np.sum(z, axis=0, keepdims=True)
        return z

    def derivative_func(
        self, ip: np.ndarray, activations: Optional[np.ndarray] = None
    ) -> np.ndarray:
        if activations is None:
            activations = self.activation_func(ip)

        categories = activations.shape[0]

        grads = np.eye(categories, dtype=np.float32) - activations.T[..., None]
        grads *= activations[:, None, :].T

        return np.moveaxis(grads, 0, -1)

    def backprop_inputs(self, grad: np.ndarray, *args, **kwargs) -> np.ndarray:
        grad = super().backprop_inputs(grad, *args, **kwargs)

        return np.sum(grad, axis=1)


class Tanh(Activation):
    """Tanh activation.

    Definition
    ----------
    tanh(ip) = (e^ip - e^(-ip)) / (e^ip + e^(-ip)), where e is Euler's constant.

    Derivative
    ----------
    tanh'(ip) = 1 - tanh(ip) ^ 2

    Input shape
    ----------
    (..., batch_size), where ... represents any number of dimensions.

    Output shape
    ----------
    Same as the input shape.

    Example
    ----------
    >>> from dnn.layers.activations import Tanh
    >>> import numpy as np
    >>> x = np.array([-2, -1, 0, 1, 2])

    >>> tanh = tanh()

    >>> tanh.compute_activations(x)
    array([-0.9640276, -0.7615942, 0., 0.7615942, 0.9640276], dtype=float32)

    >>> tanh.compute_derivatives(x)
    array([0.07065082, 0.41997433, 1., 0.41997433, 0.07065082], dtype=float32)
    """

    name = "tanh"

    def activation_func(self, ip: np.ndarray) -> np.ndarray:
        return np.tanh(ip)

    def derivative_func(
        self, ip: np.ndarray, activations: Optional[np.ndarray] = None
    ) -> np.ndarray:
        if activations is None:
            activations = self.activation_func(ip)

        activations **= 2
        return 1 - activations


class ReLU(Activation):
    name = "relu"

    def activation_func(self, ip):
        return np.maximum(0, ip)

    def derivative_func(self, ip, activations=None):
        return np.where(ip > 0, 1.0, 0)


class LeakyReLU(Activation):
    name = "lrelu"
    default_alpha = 0.01

    def __init__(self, *args, ip: Optional[LayerInput] = None, **kwargs) -> None:
        alpha = kwargs.pop("alpha", None)
        if alpha is None:
            alpha = self.default_alpha
        self.alpha = alpha

        super().__init__(ip=ip, *args, **kwargs)

    def activation_func(self, ip: np.ndarray) -> np.ndarray:
        return np.where(ip > 0, ip, self.alpha * ip)

    def derivative_func(
        self, ip: np.ndarray, activations: Optional[np.ndarray] = None
    ) -> np.ndarray:
        return np.where(ip > 0, 1.0, self.alpha)


class ELU(LeakyReLU):
    name = "elu"
    default_alpha = 1.0

    def activation_func(self, ip: np.ndarray) -> np.ndarray:
        return np.where(ip > 0, ip, self.alpha * (np.exp(ip) - 1))

    def derivative_func(
        self, ip: np.ndarray, activations: Optional[np.ndarray] = None
    ) -> np.ndarray:
        return np.where(ip > 0, 1.0, self.alpha * np.exp(ip))
