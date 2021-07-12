from math import ceil
from dnn.layers.activations import Activation
from dnn.utils import activation_factory

import numpy as np


def add_activation(activation):
    if activation is None:
        return

    if isinstance(activation, Activation):
        return activation

    return activation_factory(activation)


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
    return ((i * sH, j * sW) for i in range(oH) for j in range(oW))


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
