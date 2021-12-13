from __future__ import annotations

from abc import abstractmethod
from typing import Union

import numpy as np
from numba import njit

from .base_layer import BaseLayer, LayerInputType


@njit(cache=True, parallel=True)
def _relu(ip: np.ndarray) -> np.ndarray:
    """JIT function to compute the ReLU activation function.

    Arguments
    ----------
    ip: Input for which the activations are to be computed.
    """
    return np.maximum(0, ip)


@njit(cache=True)
def _softmax_derivative(activations: np.ndarray) -> np.ndarray:
    """JIT function to compute the derivative of softmax with
    respect to its input.

    Arguments
    ----------
    activations: Output of softmax for the input.
    """
    categories = activations.shape[0]

    # Create a (batch_size, c, c) array where each row i in the (c, c) matrices
    # has 1 - s_i in the i-th column and -s_i everywhere else
    grads = np.eye(categories).astype(np.float32) - np.expand_dims(activations.T, -1)

    # RHS converts activations from (c, batch_size) to (batch_size, 1, c)
    # The operation, thus, broadcasts each (1, c) vector to (c, c)
    # This multiplies each row in the (c, c) matrices with [s_1, s_2, ..., s_c]
    grads *= np.expand_dims(activations, 1).T

    # Move the batch_size back to the end
    axes = (1, 2, 0)
    grads = np.transpose(grads, axes=axes)
    return grads


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
        Common name for the activation. Defaults to None.
        Overriding this and setting it to a non-None value will
        ensure that the subclass is registered in the registry.

    REGISTRY: dict of (str, subclass of Activation)
        Registry of all subclasses (at any depth).

    Methods
    ---------
    activation_func(ip: np.ndarray) -> np.ndarray
        Returns the computed activations for the given input.

    derivative_func(ip: np.ndarray, activations: Optional[np.ndarray] = None) -> np.ndarray
        Returns the derivative of the activation function with respect to the given input.

    compute_activations(ip: Optional[np.ndarray] = None) -> np.ndarray
        Returns the computed activations for either the given input or the passed
        input when the class is used as a layer. In case both are available,
        the given input takes precedence. This is the method that should be used
        to do computations.

    compute_derivatives(ip: Optional[np.ndarray] = None) -> np.ndarray
        Returns the derivative of the activation function with respect to either the
        given input or the passed input when used as a layer. In case both are available,
        the given input takes precedence. This is the method that should be used
        to do computations.

    should_reshape(shape: Tuple[int, ...]) -> bool
        Returns True if the an array with the given shape should be reshaped.

    reshape(array: np.ndarray) -> np.ndarray
        Static method which reshapes the given array.

    Interface
    ----------
    Subclasses must implement the following methods:
        - activation_func(ip)
        - derivative_func(ip, activations)
    """

    name: str = None
    REGISTRY: dict[str, type[Activation]] = {}
    reset = ("activations",)

    def __init__(
        self,
        *args,
        ip: LayerInputType | None = None,
        name: str = None,
        trainable=False,
        **kwargs,
    ) -> None:
        """
        Arguments
        ---------
        ip: Input to the layer. Defaults to None, which is useful when the activation
        is not part of some model.

        name: Name of the layer (used in computation graphs). The name should be unique
        wrt all other layers in the same model. When None, a name is automatically created.
        Defaults to None.

        trainable: Indicates whether the layer is trainable or not. It is useful for
        activation functions like Maxout. Defaults to False.
        """
        super().__init__(ip=ip, trainable=trainable, name=name)

        self.activations = None

    def __init_subclass__(cls, **kwargs) -> None:
        if (name := cls.name) is not None:
            Activation.REGISTRY[name] = cls

    def should_reshape(self, shape: tuple[int, ...]) -> bool:
        """Method to determine if the given shape requires reshaping.

        By default, it returns False, implying no reshapping is required.

        Arguments
        ----------
        shape: Shape that is should be considered for reshapping.
        """
        return False

    @staticmethod
    def reshape(array: np.ndarray) -> np.ndarray:
        """Method to reshape the given NumPy array.

        By default, the array is returned as is.

        Arguments
        ----------
        array: Array to be reshapped.
        """
        return array

    @abstractmethod
    def activation_func(self, ip: np.ndarray) -> np.ndarray:
        """The formula used to calculate the activations.

        If the activation function is called g with input ip,
        this should return g(ip).

        Arguments
        ----------
        ip: The input for the function.

        Example
        ----------
        The sigmoid activation function is defined as 1 / (1 + e^(-ip)).

        This can be implemented as:
        >>> def activation_func(self, ip: np.ndarray) -> np.ndarray:
        ...     return 1 / (1 + np.exp(-ip))
        """

    @abstractmethod
    def derivative_func(
        self, ip: np.ndarray, activations: np.ndarray | None = None
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

    def compute_activations(self, ip: np.ndarray | None = None) -> np.ndarray:
        """Method to handle whether the input is being supplied externally or
        being passed by some layer, and calculate the activations accordingly.

        Arguments
        ---------
        ip: External input. When not None, this will be used to calculate the activations.
        Defaults to None.
        """
        if ip is None:
            ip = self.input()

        # Init variable to store original shape
        old_shape = None

        # Reshape the input, if required
        if self.should_reshape(ip.shape):
            old_shape = ip.shape
            ip = self.reshape(ip)

        activations = self.activation_func(ip)

        # Coerce only if necessary
        if activations.dtype != np.float32:
            activations = activations.astype(np.float32)

        # Restore the old shape so that activations has same shape
        # As the original input
        if old_shape is not None:
            activations.shape = old_shape

        return activations

    def compute_derivatives(self, ip: np.ndarray | None = None) -> np.ndarray:
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

        # Reshape the input and the activations, if required
        if self.should_reshape(ip.shape):
            ip = self.reshape(ip)

            if activations is not None:
                activations = self.reshape(activations)

        # Unlike activations, the shape is not restored since it is not necessary
        # That the derivative has the same shape as input
        derivatives = self.derivative_func(ip, activations=activations)

        # Coerce only if necessary
        if derivatives.dtype != np.float32:
            derivatives = derivatives.astype(np.float32)

        return derivatives

    def output(self) -> np.ndarray | None:
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

    Inherits from
    ----------
    Activation

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
        self, ip: np.ndarray, activations: np.ndarray | None = None
    ) -> np.ndarray:
        return np.ones_like(ip, dtype=np.float32)

    def backprop_inputs(self, grad: np.ndarray, *args, **kwargs) -> np.ndarray:
        # Just return the gradient as is
        return grad


class Sigmoid(Activation):
    """Sigmoid activation.

    Inherits from
    ----------
    Activation

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
        z = np.exp(-ip)
        z += 1
        z = 1.0 / z
        return z

    def derivative_func(
        self, ip: np.ndarray, activations: np.ndarray | None = None
    ) -> np.ndarray:
        if activations is None:
            activations = self.activation_func(ip)

        return activations * (1 - activations)


class Softmax(Activation):
    """Softmax activation.

    The softmax activation is unique since it is defined only
    for vectors. It takes a vector as input and outputs another vector.
    Its derivative is a matrix.

    Inherits from
    ----------
    Activation

    Definition
    ----------
    softmax(ip) = e^(ip) / sum(e^(ip)), where e is Euler's constant.

    Derivative
    ----------
    Consider that ip is a c-dim column vector. Then softmax(ip) is another c-dim column
    vector, [s_1, s_2, ..., s_c]. The derivate is a (c, c) matrix where the entry in
    row i and column j is defined as:
        - s_i * (1 - s_i) if i = j
        - -(s_i * s_j) if i != j

    eg - If ip is 3-dimensional, then the derivative is a 3 x 3 matrix:
    >>> [
    ...     [(1 - s_1) * s_1, -s_1 * s_2, -s_1 * s_3],
    ...     [-s_2 * s_1, (1 - s_2) * s_2, -s_2 * s_3],
    ...     [-s_3 * s_1, -s_3 * s_2, (1 - s_3) * s_3],
    ... ]

    When there is more than one vector, i.e. when ip is a (c, n) matrix,
    where c is the number of dimensions in each vector and n is the number of vectors,
    then the derivative is (c, c, n), with one (c, c) matrix for each vector.

    When the ip is (c, d1, d2, ..., n), it is reshaped into (c, d1 * d2 * .. * n).
    The derivative is (c, c, d1 * d2 * ... * n).

    Input shape
    ----------
    (c, ..., batch_size), where c is the number of categories and ...
    represents any number of dimensions.

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

    def should_reshape(self, shape: tuple[int, ...]) -> bool:
        """Method to determine if the given shape requires reshaping.

        Returns True if shape has more than two dimensions.

        Arguments
        ----------
        shape: Shape that should be considered for reshapping.
        """
        return len(shape) > 2

    @staticmethod
    def reshape(array: np.ndarray) -> np.ndarray:
        categories = array.shape[0]
        # Flatten everything except the category axis
        return array.reshape(categories, -1)

    def activation_func(self, ip: np.ndarray) -> np.ndarray:
        # Subtract maximum to prevent under/overflow
        z = ip - np.max(ip, axis=0, keepdims=True)
        z = np.exp(z)
        z /= np.sum(z, axis=0, keepdims=True)
        return z

    def derivative_func(
        self, ip: np.ndarray, activations: np.ndarray | None = None
    ) -> np.ndarray:
        if activations is None:
            activations = self.activation_func(ip)

        return _softmax_derivative(activations)

    def backprop_inputs(self, grad: np.ndarray, *args, **kwargs) -> np.ndarray:
        # Init variable to store old shape
        old_shape = None

        # It maybe that the gradient has more than two dimensions
        # This automatically implies that the input to softmax also had
        # More than two dimensions. Thus, the gradient is reshaped to
        # Account for the shape of the derivative of softmax
        if self.should_reshape(grad.shape):
            old_shape = grad.shape
            grad = self.reshape(grad)

        grad = super().backprop_inputs(grad, *args, **kwargs)
        grad = np.sum(grad, axis=1)

        # Restore old shape so that the correct shape is backpropagated
        if old_shape is not None:
            grad.shape = old_shape

        return grad


class Tanh(Activation):
    """Tanh activation.

    Inherits from
    ----------
    Activation

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

    >>> tanh = Tanh()

    >>> tanh.compute_activations(x)
    array([-0.9640276, -0.7615942, 0., 0.7615942, 0.9640276], dtype=float32)

    >>> tanh.compute_derivatives(x)
    array([0.07065082, 0.41997433, 1., 0.41997433, 0.07065082], dtype=float32)
    """

    name = "tanh"

    def activation_func(self, ip: np.ndarray) -> np.ndarray:
        return np.tanh(ip)

    def derivative_func(
        self, ip: np.ndarray, activations: np.ndarray | None = None
    ) -> np.ndarray:
        if activations is None:
            activations = self.activation_func(ip)

        activations **= 2
        return 1 - activations


class ReLU(Activation):
    """ReLU (Rectified Linear Unit) activation.

    Inherits from
    ----------
    Activation

    Definition
    ----------
    relu(ip) is:
        - ip if ip > 0
        - 0 if ip <= 0

    Derivative
    ----------
    relu'(ip) is:
        - 1 if ip > 0
        - 0 if ip <= 0

    Input shape
    ----------
    (..., batch_size), where ... represents any number of dimensions.

    Output shape
    ----------
    Same as the input shape.

    Example
    ----------
    >>> from dnn.layers.activations import ReLU
    >>> import numpy as np
    >>> x = np.array([-2, -1, 0, 1, 2])

    >>> relu = ReLU()

    >>> relu.compute_activations(x)
    array([0., 0., 0., 1., 2.], dtype=float32)

    >>> relu.compute_derivatives(x)
    array([0., 0., 0., 1., 1.], dtype=float32)
    """

    name = "relu"

    def activation_func(self, ip):
        return _relu(ip)

    def derivative_func(self, ip, activations=None):
        return np.where(ip > 0, 1.0, 0)


class LeakyReLU(Activation):
    """LeakyReLU (Leaky Rectified Linear Unit) activation.

    Inherits from
    ----------
    Activation

    Attributes
    ----------
    alpha: float
        Hyperparameter by which the input is multiplied for negative quantities.

    default_alpha: float
        Class attribute which is the alpha value that should be used when alpha
        is not passed during instance creation. Defaults to 0.01.

    Definition
    ----------
    lrelu(ip) is:
        - ip if ip > 0
        - alpha * ip if ip <= 0,
    where alpha is a hyperparameter

    Derivative
    ----------
    lrelu'(ip) is:
        - 1 if ip > 0
        - alpha if ip <= 0

    Input shape
    ----------
    (..., batch_size), where ... represents any number of dimensions.

    Output shape
    ----------
    Same as the input shape.

    Example
    ----------
    >>> from dnn.layers.activations import LeakyReLU
    >>> import numpy as np
    >>> x = np.array([-2, -1, 0, 1, 2])

    >>> lrelu = LeakyReLU()  # alpha = 0.01

    >>> lrelu.compute_activations(x)
    array([-0.02, -0.01, 0., 1., 2.], dtype=float32)

    >>> lrelu.compute_derivatives(x)
    array([0.01, 0.01, 0.01, 1., 1.], dtype=float32)
    """

    name = "lrelu"
    default_alpha = 0.01

    def __init__(self, *args, ip: LayerInputType | None = None, **kwargs) -> None:
        alpha = kwargs.pop("alpha", None)
        if alpha is None:
            alpha = self.default_alpha
        self.alpha = alpha

        super().__init__(ip=ip, *args, **kwargs)

    def activation_func(self, ip: np.ndarray) -> np.ndarray:
        return np.where(ip > 0, ip, self.alpha * ip)

    def derivative_func(
        self, ip: np.ndarray, activations: np.ndarray | None = None
    ) -> np.ndarray:
        return np.where(ip > 0, 1.0, self.alpha)


class ELU(LeakyReLU):
    """ELU (Exponential Linear Unit) activation.

    Inherits from
    ----------
    LeakyReLU

    Attributes
    ----------
    alpha: float
        Hyperparameter by which the input is multiplied for negative quantities.

    default_alpha: float
        Class attribute which is the alpha value that should be used when alpha
        is not passed during instance creation. Defaults to 1.0.

    Definition
    ----------
    elu(ip) is:
        - ip if ip > 0
        - alpha * (e ^ ip - 1) if ip <= 0,
    where alpha is a hyperparameter

    Derivative
    ----------
    elu'(ip) is:
        - 1 if ip > 0
        - alpha * (e ^ ip) if ip <= 0

    Input shape
    ----------
    (..., batch_size), where ... represents any number of dimensions.

    Output shape
    ----------
    Same as the input shape.

    Example
    ----------
    >>> from dnn.layers.activations import ELU
    >>> import numpy as np
    >>> x = np.array([-2, -1, 0, 1, 2])

    >>> elu = ELU()  # alpha = 1.0

    >>> elu.compute_activations(x)
    array([-0.86466473, -0.63212055, 0., 1., 2.], dtype=float32)

    >>> elu.compute_derivatives(x)
    array([0.13533528, 0.36787945, 1. , 1., 1.], dtype=float32)
    """

    name = "elu"
    default_alpha = 1.0

    def activation_func(self, ip: np.ndarray) -> np.ndarray:
        return np.where(ip > 0, ip, self.alpha * (np.exp(ip) - 1))

    def derivative_func(
        self, ip: np.ndarray, activations: np.ndarray | None = None
    ) -> np.ndarray:
        return np.where(ip > 0, 1.0, self.alpha * np.exp(ip))


# Type alias that can be used by layers to annonate activations
ActivationType = Union[Activation, str, None]
