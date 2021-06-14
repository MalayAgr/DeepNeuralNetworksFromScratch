from .activations import Activation
from .loss import Loss


def activation_factory(activation, ip=None, *args, **kwargs):
    registry = Activation._get_activation_classes()
    cls = registry.get(activation)
    if cls is None:
        raise ValueError("Activation with this name does not exist")
    return cls(ip=ip, *args, **kwargs)


def loss_factory(loss, Y, *args, **kwargs):
    registry = Loss._get_loss_classes()
    cls = registry.get(loss)
    if cls is None:
        raise ValueError("Loss with this name does not exist")
    return cls(Y, *args, **kwargs)
