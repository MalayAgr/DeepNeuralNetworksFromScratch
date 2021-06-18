from .activations import Activation
from .loss import Loss
import numpy as np


def activation_factory(activation, *args, ip=None, **kwargs):
    registry = Activation.get_activation_classes()
    cls = registry.get(activation)
    if cls is None:
        raise ValueError("Activation with this name does not exist")
    return cls(ip=ip, *args, **kwargs)


def loss_factory(loss, Y, *args, **kwargs):
    registry = Loss.get_loss_classes()
    cls = registry.get(loss)
    if cls is None:
        raise ValueError("Loss with this name does not exist")
    return cls(Y=Y, *args, **kwargs)


def generate_batches(X, Y, batch_size, shuffle=True):
    num_samples = X.shape[-1]
    num_batches = int(np.ceil(num_samples / batch_size))

    if shuffle is True:
        perm = np.random.permutation(num_samples)
        X, Y = X[:, perm], Y[:, perm]

    if num_batches == 1:
        yield X, Y, num_samples
        return

    start = 0
    for idx in range(num_batches):
        end = (idx + 1) * batch_size

        if end > num_samples:
            yield X[:, start:], Y[:, start:], num_samples - start
        else:
            yield X[:, start:end], Y[:, start:end], batch_size

        start = end
