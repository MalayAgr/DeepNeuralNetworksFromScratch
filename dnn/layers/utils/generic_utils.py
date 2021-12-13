from __future__ import annotations

from dnn.layers.activations import Activation, ActivationType
from dnn.layers.base_layer import LayerInputType


def activation_factory(
    activation: str, *args, ip: LayerInputType | None = None, **kwargs
) -> Activation:
    registry = Activation.REGISTRY
    cls = registry.get(activation)
    if cls is None:
        raise ValueError("Activation with this name does not exist.")
    return cls(ip=ip, *args, **kwargs)


def add_activation(activation: ActivationType) -> Activation:
    if isinstance(activation, Activation):
        return activation

    if activation is None:
        activation = "linear"

    return activation_factory(activation)
