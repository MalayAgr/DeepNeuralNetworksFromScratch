from dnn.layers.base_layer import LayerInput
from typing import Optional, Union
from dnn.layers.activations import Activation


def activation_factory(
    activation: str, *args, ip: Optional[LayerInput] = None, **kwargs
) -> Activation:
    registry = Activation.get_activation_classes()
    cls = registry.get(activation)
    if cls is None:
        raise ValueError("Activation with this name does not exist")
    return cls(ip=ip, *args, **kwargs)


def add_activation(activation: Union[Activation, str, None]) -> Activation:
    if isinstance(activation, Activation):
        return activation

    if activation is None:
        activation = "linear"

    return activation_factory(activation)
