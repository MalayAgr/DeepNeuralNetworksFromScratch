from math import ceil
from typing import Generator, Optional, Tuple, Union

import numpy as np

from .activations import Activation
from .base_layer import LayerInput


def activation_factory(
    activation: str, *args, ip: Optional[LayerInput] = None, **kwargs
) -> Activation:
    registry = Activation.get_activation_classes()
    cls = registry.get(activation)
    if cls is None:
        raise ValueError("Activation with this name does not exist")
    return cls(ip=ip, *args, **kwargs)


def add_activation(activation: Union[Activation, str, None]) -> Activation:
    if activation is None:
        return

    if isinstance(activation, Activation):
        return activation

    return activation_factory(activation)


def compute_conv_padding(
    kernel_size: Tuple[int, int], mode: str = "valid"
) -> Tuple[int, int]:
    if mode == "same":
        kH, kW = kernel_size
        return ceil((kH - 1) / 2), ceil((kW - 1) / 2)
    return 0, 0


def compute_conv_output_dim(n: int, f: int, p: int, s: int) -> int:
    return int((n - f + 2 * p) / s + 1)


def pad(X: np.ndarray, pad_H: int, pad_W: int) -> np.ndarray:
    return np.pad(X, ((0, 0), (pad_H, pad_H), (pad_W, pad_W), (0, 0)))


def slice_idx_generator(
    oH: int, oW: int, sH: int, sW: int
) -> Generator[Tuple[int, int], None, None]:
    return ((i * sH, j * sW) for i in range(oH) for j in range(oW))


def vectorize_for_conv(
    X: np.ndarray,
    kernel_size: Tuple[int, int],
    stride: Tuple[int, int],
    output_size: Tuple[int, int],
    reshape: Optional[Tuple] = None,
) -> np.ndarray:
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


def vectorize_kernel_for_conv(kernel, reshape=None) -> np.ndarray:
    if reshape is None:
        filters = kernel.shape[-1]
        reshape = (-1, filters)
    return kernel.reshape(*reshape)


def _prepare_ip_for_conv(
    X: np.ndarray,
    kernel_size: Tuple[int, int],
    stride: Tuple[int, int] = (1, 1),
    padding: Tuple[int, int] = (0, 0),
    vec_reshape: Optional[Tuple] = None,
) -> Tuple[np.ndarray, int, int]:

    ipH, ipW = X.shape[1:-1]
    kH, kW = kernel_size
    sH, sW = stride
    pH, pW = padding

    oH = compute_conv_output_dim(ipH, kH, pH, sH)
    oW = compute_conv_output_dim(ipW, kW, pW, sW)

    if padding != (0, 0):
        X, _ = pad(X, pH, pW)

    X = vectorize_for_conv(X, (kH, kW), stride, (oH, oW), reshape=vec_reshape)

    return X, oH, oW


def convolve(X, weights):
    return np.matmul(X, weights[None, ...], dtype=np.float32)


def convolve2d(
    X: np.ndarray,
    kernel: np.ndarray,
    stride: Tuple[int, int] = (1, 1),
    padding: Tuple[int, int] = (0, 0),
    return_vec_ip: bool = False,
    return_vec_kernel: bool = False,
) -> Union[
    Tuple[np.ndarray, np.ndarray, np.ndarray],
    Tuple[np.ndarray, np.ndarray],
    Tuple[np.ndarray],
]:
    kH, kW, filters = kernel.shape[1:]

    X, oH, oW = _prepare_ip_for_conv(
        X=X, kernel_size=(kH, kW), stride=stride, padding=padding
    )

    X = np.moveaxis(X, -1, 0)

    weights = vectorize_kernel_for_conv(kernel)

    convolution = convolve(X, weights)

    shape = (filters, oH, oW, -1)

    ret_val = (np.swapaxes(convolution, 0, 2).reshape(shape),)

    if return_vec_ip is True:
        ret_val += (X,)

    if return_vec_kernel is True:
        ret_val += (weights,)

    return ret_val


def depthwise_convolve2d(
    X: np.ndarray,
    kernel: np.ndarray,
    stride: Tuple[int, int] = (1, 1),
    padding: Tuple[int, int] = (0, 0),
    return_vec_ip: bool = False,
    return_vec_kernel: bool = False,
) -> Union[
    Tuple[np.ndarray, np.ndarray, np.ndarray],
    Tuple[np.ndarray, np.ndarray],
    Tuple[np.ndarray],
]:
    ipC, kH, kW, multiplier = kernel.shape

    X, oH, oW = _prepare_ip_for_conv(
        X=X,
        kernel_size=(kH, kW),
        stride=stride,
        padding=padding,
        vec_reshape=(ipC, (kH * kW), -1),
    )

    X = X.transpose(-1, 1, 0, 2)

    weights = vectorize_kernel_for_conv(kernel, reshape=(ipC, -1, multiplier))

    convolution = convolve(X, weights)

    shape = (ipC * multiplier, oH, oW, -1)

    ret_val = (np.swapaxes(convolution, 0, -1).reshape(shape),)

    if return_vec_ip is True:
        ret_val += (X,)

    if return_vec_kernel is True:
        ret_val += (weights,)

    return ret_val


def accumulate_dX_conv(
    dX_shape: Tuple,
    output_size: Tuple[int, int],
    dIp: np.ndarray,
    stride: Tuple[int, int],
    kernel_size: Tuple[int, int],
    reshape: Tuple,
    padding: Tuple[int, int] = (0, 0),
    moveaxis: bool = True,
) -> np.ndarray:
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
