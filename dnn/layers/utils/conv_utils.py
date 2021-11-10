from typing import Generator, Tuple

from math import ceil

import numpy as np


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


def vectorize_ip_for_conv(
    X: np.ndarray,
    kernel_size: Tuple[int, int],
    stride: Tuple[int, int],
    output_size: Tuple[int, int],
    reshape: Tuple[int, ...] = (),
) -> np.ndarray:
    sH, sW = stride
    kH, kW = kernel_size
    oH, oW = output_size

    indices = slice_idx_generator(oH, oW, sH, sW)

    if not reshape:
        reshape = (-1, X.shape[-1])

    vectorized_ip = np.array(
        tuple(X[:, i : i + kH, j : j + kW].reshape(*reshape) for i, j in indices)
    )

    return vectorized_ip


def vectorize_kernel_for_conv(
    kernel: np.ndarray, reshape: Tuple[int, ...] = ()
) -> np.ndarray:
    if not reshape:
        filters = kernel.shape[-1]
        reshape = (-1, filters)
    return kernel.reshape(*reshape)


def prepare_ip_for_conv(
    X: np.ndarray,
    kernel_size: Tuple[int, int],
    stride: Tuple[int, int] = (1, 1),
    padding: Tuple[int, int] = (0, 0),
    vec_reshape: Tuple[int, ...] = (),
) -> np.ndarray:

    ipH, ipW = X.shape[1:-1]
    kH, kW = kernel_size
    sH, sW = stride
    pH, pW = padding

    oH = compute_conv_output_dim(ipH, kH, pH, sH)
    oW = compute_conv_output_dim(ipW, kW, pW, sW)

    if padding != (0, 0):
        X, _ = pad(X, pH, pW)

    return vectorize_ip_for_conv(X, (kH, kW), stride, (oH, oW), reshape=vec_reshape)


def convolve(X, weights):
    return np.matmul(X, weights[None, ...], dtype=np.float32)


def convolve2d(
    X: np.ndarray, weights: np.ndarray, filters: int, op_area: Tuple[int, int]
) -> np.ndarray:
    oH, oW = op_area

    convolution = convolve(X, weights)

    shape = (filters, oH, oW, -1)

    convolution = np.swapaxes(convolution, 0, 2).reshape(shape)

    return convolution


def depthwise_convolve2d(
    X: np.ndarray,
    weights: np.ndarray,
    multiplier: int,
    ip_C: int,
    op_area: Tuple[int, int],
) -> np.ndarray:
    oH, oW = op_area

    convolution = convolve(X, weights)

    shape = (multiplier * ip_C, oH, oW, -1)

    convolution = np.moveaxis(convolution, [0, -1], [-1, 1]).reshape(shape)

    return convolution


def _backprop_kernel_generic_conv(
    ip: np.ndarray,
    grad: np.ndarray,
    kernel_size: Tuple[int, int],
    filters: int,
    axis: Tuple[int, ...] = (0,),
) -> np.ndarray:
    dW = np.matmul(ip, grad, dtype=np.float32)
    dW = dW.sum(axis=axis)
    kH, kW = kernel_size
    return dW.reshape(-1, kH, kW, filters)


def backprop_kernel_conv2d(
    ip: np.ndarray, grad: np.ndarray, kernel_size: Tuple[int, int], filters: int
) -> np.ndarray:
    return _backprop_kernel_generic_conv(
        ip=ip[..., None],
        grad=grad[..., None, :],
        kernel_size=kernel_size,
        filters=filters,
        axis=(0, 1),
    )


def backprop_kernel_depthwise_conv2d(
    ip: np.ndarray, grad: np.ndarray, kernel_size: Tuple[int, int], multiplier: int
) -> np.ndarray:
    return _backprop_kernel_generic_conv(
        ip=np.swapaxes(ip, -1, -2),
        grad=grad,
        kernel_size=kernel_size,
        filters=multiplier,
    )


def backprop_bias_conv(
    grad: np.ndarray, axis: Tuple, reshape: Tuple[int, ...] = ()
) -> np.ndarray:
    grad = grad.sum(axis=axis)
    if reshape:
        grad = grad.reshape(-1, *reshape)
    return grad


def backprop_ip_conv2d(grad: np.ndarray, kernel: np.ndarray) -> np.ndarray:
    return np.matmul(grad, kernel.T, dtype=np.float32)


def backprop_ip_depthwise_conv2d(grad: np.ndarray, kernel: np.ndarray) -> np.ndarray:
    kernel = np.swapaxes(kernel, -1, -2)
    dIp = np.matmul(grad, kernel, dtype=np.float32)
    return np.moveaxis(dIp, 2, 1)


def accumulate_dX_conv(
    dX_shape: Tuple[int, ...],
    output_size: Tuple[int, int],
    dIp: np.ndarray,
    stride: Tuple[int, int],
    kernel_size: Tuple[int, int],
    reshape: Tuple[int, ...],
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
