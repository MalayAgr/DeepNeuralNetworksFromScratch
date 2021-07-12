import functools
from math import ceil

import numpy as np

from dnn.activations import Activation
from dnn.loss import Loss


def activation_factory(activation, *args, ip=None, **kwargs):
    registry = Activation.get_activation_classes()
    cls = registry.get(activation)
    if cls is None:
        raise ValueError("Activation with this name does not exist")
    return cls(ip=ip, *args, **kwargs)


def loss_factory(loss):
    registry = Loss.get_loss_classes()
    cls = registry.get(loss)
    if cls is None:
        raise ValueError("Loss with this name does not exist")
    return cls()


def add_activation(activation):
    if activation is None:
        return

    if isinstance(activation, Activation):
        return activation

    return activation_factory(activation)


def generate_batches(X, Y, batch_size, shuffle=True):
    num_samples = X.shape[-1]

    if batch_size > num_samples:
        raise ValueError(
            "The batch size is greater than the number of samples in the dataset"
        )

    num_full_batches = int(np.floor(num_samples / batch_size))

    if shuffle is True:
        perm = np.random.permutation(num_samples)
        X, Y = X[..., perm], Y[..., perm]

    if num_full_batches == 1:
        yield X, Y, num_samples
        return

    for idx in range(num_full_batches):
        start = idx * batch_size
        end = (idx + 1) * batch_size
        yield X[..., start:end], Y[..., start:end], batch_size

    if num_samples % batch_size != 0:
        start = batch_size * num_full_batches
        yield X[..., start:], Y[..., start:], num_samples - start


def backprop(model, loss, labels, preds, reg_param=0.0):
    dA = loss.compute_derivatives(labels, preds)

    for layer in reversed(model.layers):
        dA = layer.backprop_step(dA, reg_param=reg_param)


def compute_l2_cost(model, reg_param, cost):
    norm = np.add.reduce([np.linalg.norm(layer.weights) ** 2 for layer in model.layers])

    m = model.ip_layer.ip.shape[-1]

    return cost + (reg_param * norm) / (2 * m)


def rgetattr(obj, attr, *args):
    def _getattr(obj, attr):
        return getattr(obj, attr, *args)

    return functools.reduce(_getattr, [obj] + attr.split("."))


def rsetattr(obj, attr, val):
    pre, _, post = attr.rpartition(".")
    return setattr(rgetattr(obj, pre) if pre else obj, post, val)


def compute_conv_padding(kernel_size, mode="valid"):
    if mode == "same":
        kH, kW = kernel_size
        return ceil((kH - 1) / 2), ceil((kW - 1) / 2)
    return 0, 0


def compute_conv_output_dim(n, f, p, s):
    return int((n - f + 2 * p) / s + 1)


def pad(X, pad_H, pad_W):
    padded = np.pad(X, ((0, 0), (pad_H, pad_H), (pad_W, pad_W), (0, 0)))

    new_size = padded.shape[1], padded.shape[2]

    return padded, new_size


def slice_idx_generator(oH, oW, sH, sW):
    return ((i * sH, j * sW) for i in range(oH) for j in range(oH))


def vectorize_for_conv(X, kernel_size, stride, output_size, reshape=None):
    sH, sW = stride
    kH, kW = kernel_size
    oH, oW = output_size

    indices = slice_idx_generator(oH, oW, sH, sW)

    if reshape is None:
        reshape = (-1, X.shape[-1])

    vectorized_ip = np.array(
        [X[:, i : i + kH, j : j + kW].reshape(*reshape) for i, j in indices]
    )

    return vectorized_ip


def accumulate_dX_conv(
    dX_shape,
    output_size,
    dIp,
    stride,
    kernel_size,
    reshape,
    padding=(0, 0),
    moveaxis=True,
):
    kH, kW = kernel_size
    sH, sW = stride

    dX = np.zeros(shape=dX_shape, dtype=np.float32)

    slice_idx = slice_idx_generator(output_size[0], output_size[1], sH, sW)

    for idx, (start_r, start_c) in enumerate(slice_idx):
        end_r, end_c = start_r + kH, start_c + kW
        dX[..., start_r:end_r, start_c:end_c] += dIp[:, idx, ...].reshape(*reshape)

    if padding != (0, 0):
        pH, pW = padding
        dX = dX[..., pH:-pH, pW:-pW]

    if moveaxis is False:
        return dX

    return np.moveaxis(dX, 0, -1)
