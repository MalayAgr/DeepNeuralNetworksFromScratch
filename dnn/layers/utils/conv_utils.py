from __future__ import annotations

import math

import numpy as np
from numba import njit


@njit(cache=True)
def _slice_idx_generator(oH: int, oW: int, sH: int, sW: int) -> np.ndarray:
    indices = np.empty((oH * oW, 2), np.int16)

    ax = np.arange(oH)
    ax *= sH
    ax = np.repeat(ax, oW)

    indices[:, 0] = ax

    ax = np.arange(oW)
    ax *= sW
    ax = np.repeat(ax, oH)
    ax = ax.reshape((oW, oH))
    ax = ax.T.flatten()

    indices[:, 1] = ax

    return indices


@njit(cache=True)
def _vectorize_ip_reshape(
    X: np.ndarray,
    kernel_size: tuple[int, int],
    stride: tuple[int, int],
    output_size: tuple[int, int],
    reshape: tuple[int, ...],
    dim: int = 0,
) -> np.ndarray:
    sH, sW = stride
    kH, kW = kernel_size
    oH, oW = output_size
    batch_size = X.shape[-1]

    indices = _slice_idx_generator(oH, oW, sH, sW)
    n_indices = indices.shape[0]

    if dim == 0:
        dim = reshape[0]

    areas = np.empty((n_indices, dim, kH, kW, batch_size), np.float32)

    idx = 0
    for i, j in indices:
        areas[idx, ...] = X[:, i : i + kH, j : j + kW]
        idx += 1

    areas = areas.reshape((n_indices, *reshape))
    return areas


@njit(cache=True)
def _vectorize_ip_no_reshape(
    X: np.ndarray,
    kernel_size: tuple[int, int],
    stride: tuple[int, int],
    output_size: tuple[int, int],
) -> np.ndarray:
    shape = X.shape
    filters, batch_size = shape[0], shape[-1]

    return _vectorize_ip_reshape(
        X=X,
        kernel_size=kernel_size,
        stride=stride,
        output_size=output_size,
        reshape=(-1, batch_size),
        dim=filters,
    )


@njit(cache=True)
def _vectorize_kernel_reshape(
    kernel: np.ndarray, reshape: tuple[int, ...]
) -> np.ndarray:
    return kernel.reshape(reshape)


@njit(cache=True)
def _vectorize_kernel_no_reshape(kernel: np.ndarray) -> np.ndarray:
    return _vectorize_kernel_reshape(kernel=kernel, reshape=(-1, kernel.shape[-1]))


def _backprop_kernel(
    ip: np.ndarray,
    grad: np.ndarray,
    kernel_size: tuple[int, int],
    filters: int,
    axis: tuple[int, ...] = (0,),
) -> np.ndarray:
    dW = np.matmul(ip, grad, dtype=np.float32)
    dW = dW.sum(axis=axis)
    kH, kW = kernel_size
    return dW.reshape(-1, kH, kW, filters)


@njit(cache=True)
def padding(kernel_size: tuple[int, int], mode: str = "valid") -> tuple[int, int]:
    if mode == "same":
        kH, kW = kernel_size
        return math.ceil((kH - 1) / 2), math.ceil((kW - 1) / 2)
    return 0, 0


def convolution_output_dim(n: int, f: int, p: int, s: int) -> int:
    return math.floor((n - f + 2 * p) / s + 1)


def deconvolution_output_dim(n: int, f: int, p: int, s: int) -> int:
    return (n - 1) * s + f - 2 * p


def pad(X: np.ndarray, pad_H: int, pad_W: int) -> np.ndarray:
    return np.pad(X, ((0, 0), (pad_H, pad_H), (pad_W, pad_W), (0, 0)))


def prepare_ip(
    X: np.ndarray,
    kernel_size: tuple[int, int],
    stride: tuple[int, int] = (1, 1),
    padding: tuple[int, int] = (0, 0),
    vec_reshape: tuple[int, ...] = (),
) -> np.ndarray:

    ipH, ipW = X.shape[1:-1]
    kH, kW = kernel_size
    sH, sW = stride
    pH, pW = padding

    oH = convolution_output_dim(ipH, kH, pH, sH)
    oW = convolution_output_dim(ipW, kW, pW, sW)

    if padding != (0, 0):
        X = pad(X, pH, pW)

    return (
        _vectorize_ip_no_reshape(X, kernel_size, stride, (oH, oW))
        if not vec_reshape
        else _vectorize_ip_reshape(
            X, kernel_size, stride, (oH, oW), reshape=vec_reshape
        )
    )


def vectorize_kernel(kernel: np.ndarray, reshape: tuple[int, ...] = ()) -> np.ndarray:
    return (
        _vectorize_kernel_no_reshape(kernel)
        if not reshape
        else _vectorize_kernel_reshape(kernel, reshape=reshape)
    )


def convolve(X: np.ndarray, weights: np.ndarray) -> np.ndarray:
    return np.matmul(X, weights[None, ...], dtype=np.float32)


@njit(cache=True)
def accumulate_dX_conv(
    grad_shape: tuple[int, ...],
    output_size: tuple[int, int],
    vec_ip_grad: np.ndarray,
    stride: tuple[int, int],
    kernel_size: tuple[int, int],
    reshape: tuple[int, ...],
    padding: tuple[int, int] = (0, 0),
    moveaxis: bool = True,
) -> np.ndarray:
    kH, kW = kernel_size
    sH, sW = stride

    grad = np.zeros(shape=grad_shape, dtype=np.float32)

    slice_idx = _slice_idx_generator(output_size[0], output_size[1], sH, sW)

    idx = 0
    for start_r, start_c in slice_idx:
        end_r, end_c = start_r + kH, start_c + kW
        area = np.ascontiguousarray(vec_ip_grad[:, idx, ...])
        grad[..., start_r:end_r, start_c:end_c] += area.reshape(reshape)
        idx += 1

    if padding != (0, 0):
        pH, pW = padding
        grad = grad[..., pH:-pH, pW:-pW]

    if moveaxis is True:
        axes = (1, 2, 3, 0)
        grad = np.transpose(grad, axes)

    return grad


def backprop_kernel_conv2d(
    ip: np.ndarray, grad: np.ndarray, kernel_size: tuple[int, int], filters: int
) -> np.ndarray:
    return _backprop_kernel(
        ip=ip[..., None],
        grad=grad[..., None, :],
        kernel_size=kernel_size,
        filters=filters,
        axis=(0, 1),
    )


def backprop_kernel_depthwise_conv2d(
    ip: np.ndarray, grad: np.ndarray, kernel_size: tuple[int, int], multiplier: int
) -> np.ndarray:
    return _backprop_kernel(
        ip=np.swapaxes(ip, -1, -2),
        grad=grad,
        kernel_size=kernel_size,
        filters=multiplier,
    )


def backprop_bias(
    grad: np.ndarray, axis: tuple[int, ...], reshape: tuple[int, ...]
) -> np.ndarray:
    grad = grad.sum(axis=axis)
    grad = grad.reshape((-1, *reshape))
    return grad


def convolve2d(
    X: np.ndarray, weights: np.ndarray, filters: int, op_area: tuple[int, int]
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
    op_area: tuple[int, int],
) -> np.ndarray:
    oH, oW = op_area

    convolution = convolve(X, weights)

    shape = (multiplier * ip_C, oH, oW, -1)

    convolution = np.moveaxis(convolution, [0, -1], [-1, 1]).reshape(shape)

    return convolution


def transpose_convolve2d(
    X: np.ndarray,
    weights: np.ndarray,
    shape: tuple[int, ...],
    filters: int,
    ip_area: tuple[int, int],
    kernel_size: tuple[int, int],
    stride: tuple[int, int],
    padding: tuple[int, int],
) -> np.ndarray:
    elem_wise = np.matmul(X, weights.T, dtype=np.float32)

    return accumulate_dX_conv(
        grad_shape=shape,
        output_size=ip_area,
        vec_ip_grad=elem_wise,
        stride=stride,
        kernel_size=kernel_size,
        padding=padding,
        reshape=(-1, filters, *kernel_size),
    )


@njit(cache=True)
def maxpool2D(
    X: np.ndarray, windows: int, output_size: tuple[int, int]
) -> tuple[np.ndarray, np.ndarray]:
    first_three = X.shape[:-1]

    maximums = np.empty(first_three, np.float32)
    mask = np.zeros_like(X, np.bool_)

    for i, j, k in np.ndindex(first_three):
        pos = X[i, j, k, :].argmax()
        maximums[i, j, k] = X[i, j, k, pos]
        mask[i, j, k, pos] = True

    shape = (-1, *output_size, windows)
    maximums = maximums.reshape(shape)
    maximums = np.transpose(maximums, (3, 1, 2, 0))
    return maximums, mask


def averagepool2D(
    X: np.ndarray, windows: int, output_size: tuple[int, int]
) -> tuple[np.ndarray, np.ndarray]:
    shape = X.shape
    pool_size = X.shape[-1]

    averages = X.mean(axis=-1)

    distributed = np.ones(shape=(1, 1, 1, pool_size), dtype=np.float32)
    distributed /= pool_size

    shape = (windows, *output_size, -1)

    return np.swapaxes(averages, 0, -1).reshape(*shape), distributed
