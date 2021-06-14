from .activations import Activation


def activation_factory(activation, ip=None, *args, **kwargs):
    registry = Activation._get_activation_classes()
    cls = registry.get(activation)
    if cls is None:
        raise ValueError("Activation with this name does not exist")
    return cls(ip=ip, *args, **kwargs)
