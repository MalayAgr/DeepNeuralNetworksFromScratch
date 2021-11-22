from __future__ import annotations

import math
from collections.abc import Iterator
from typing import List, Tuple

import numpy as np
from numba import njit

from dnn.loss import Loss

BatchGenerator = Iterator[Tuple[np.ndarray, np.ndarray, int]]
UnpackReturnType = Iterator[Tuple[List[np.ndarray], List[np.ndarray], List[int]]]


def loss_factory(loss: str) -> Loss:
    registry = Loss.REGISTRY
    cls = registry.get(loss)
    if cls is None:
        raise ValueError("Loss with this name does not exist")
    return cls()


@njit
def generate_batches(
    X: np.ndarray, Y: np.ndarray, batch_size: int, shuffle: bool = True
) -> BatchGenerator:
    num_samples = X.shape[-1]

    if batch_size > num_samples:
        msg = "The batch size is greater than the number of samples in the dataset."
        raise ValueError(msg)

    num_full_batches = math.floor(num_samples / batch_size)

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


def _unpack_data_generators(generators: Tuple[BatchGenerator]) -> UnpackReturnType:
    for input_batches in zip(*generators):
        batch_X, batch_Y, sizes = [], [], []
        for X, Y, size in input_batches:
            batch_X.append(X)
            batch_Y.append(Y)
            sizes.append(size)
        yield batch_X, batch_Y, sizes


def get_data_generator(
    X: List[np.ndarray], Y: List[np.ndarray], batch_size: int, shuffle: bool = True
) -> UnpackReturnType:
    generators = tuple(
        generate_batches(x, y, batch_size=batch_size, shuffle=shuffle)
        for x, y in zip(X, Y)
    )
    return _unpack_data_generators(generators=generators)
