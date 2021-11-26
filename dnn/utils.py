from __future__ import annotations

import functools
import itertools
import math
from collections.abc import Iterator
from typing import Tuple

import numpy as np
from numba import njit

from dnn.loss import Loss

BatchIterator = Iterator[np.ndarray]
DatasetIterator = Iterator[Tuple[Tuple[np.ndarray], Tuple[np.ndarray], int]]


def loss_factory(loss: str) -> Loss:
    registry = Loss.REGISTRY
    cls = registry.get(loss)
    if cls is None:
        raise ValueError("Loss with this name does not exist")
    return cls()


@njit
def _batches_without_permutation(
    X: np.ndarray,
    batch_size: int,
) -> BatchIterator:
    num_samples = X.shape[-1]

    num_full_batches = math.floor(num_samples / batch_size)

    if num_full_batches == 1:
        yield X
        return

    for idx in range(num_full_batches):
        start = idx * batch_size
        end = (idx + 1) * batch_size
        yield X[..., start:end]

    if num_samples % batch_size != 0:
        start = batch_size * num_full_batches
        yield X[..., start:]


@njit
def _batches_with_permutation(
    X: np.ndarray, batch_size: int, perm: np.ndarray
) -> BatchIterator:
    X = X[..., perm]
    return _batches_without_permutation(X, batch_size)


def get_batch_generator(
    X: Tuple[np.ndarray], Y: Tuple[np.ndarray], batch_size: int, shuffle: bool = True
) -> DatasetIterator:
    num_samples = X[0].shape[-1]

    if batch_size > num_samples:
        msg = "The batch size is greater than the number of samples in the dataset."
        raise ValueError(msg)

    if shuffle is True:
        perm = np.random.permutation(num_samples)
        func = functools.partial(
            _batches_with_permutation, batch_size=batch_size, perm=perm
        )
    else:
        func = functools.partial(_batches_without_permutation, batch_size=batch_size)

    gens = (func(X=array) for array in itertools.chain(X, Y))

    X_len = len(X)

    for batches in zip(*gens):
        size = batches[0].shape[-1]
        yield batches[:X_len], batches[X_len:], size


class HeightWidthAttribute:
    def __init__(self, height_attr="", width_attr="") -> None:
        self.height_attr = height_attr
        self.width_attr = width_attr

    def __set_name__(self, owner, name: str):
        self.private_name = "_" + name
        self.height_attr = self.height_attr if self.height_attr else f"{name}_H"
        self.width_attr = self.width_attr if self.width_attr else f"{name}_W"

    def __get__(self, obj, klass=None):
        return getattr(obj, self.private_name)

    def __set__(self, obj, value: Tuple[int, int]):
        setattr(obj, self.private_name, value)
        setattr(obj, self.height_attr, value[0])
        setattr(obj, self.width_attr, value[1])
