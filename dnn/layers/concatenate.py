from typing import Generator, List, Optional, Tuple

import numpy as np

from .base_layer import LayerInput, MultiInputBaseLayer


class Concatenate(MultiInputBaseLayer):
    def __init__(self, ip: List[LayerInput], axis: int = 0, name: str = None) -> None:
        super().__init__(ip, trainable=False, name=name)

        self._validate_same_shape(axis)

        self.axis = axis
        self.concatenated: np.ndarray

    def _get_axis_excluded_shapes(self, axis: int) -> Generator[Tuple, None, None]:
        return (
            tuple(dim for i, dim in enumerate(shape) if i != axis)
            for shape in self.input_shape()
        )

    def _validate_same_shape(self, axis: int) -> None:
        shapes = self._get_axis_excluded_shapes(axis)
        first_shape = next(shapes)
        if any(shape != first_shape for shape in shapes):
            msg = (
                "All inputs must output the same shape, except along the concatenation axis. "
                f"Expected shape (excluding the concatenation axis): {first_shape}."
            )
            raise ValueError(msg)

    def output(self) -> Optional[np.ndarray]:
        return self.concatenated

    def output_shape(self) -> Tuple:
        axis = self.axis

        shapes = self.input_shape()

        first_shape = shapes[0]
        in_axis_dim = first_shape[axis]

        if in_axis_dim is None:
            return first_shape

        out_axis_dim = sum((shape[axis] for shape in shapes[1:]), in_axis_dim)
        shape = list(first_shape)
        shape[axis] = out_axis_dim

        return tuple(shape)

    def forward_step(self, *args, **kwargs) -> np.ndarray:
        self._validate_same_shape(self.axis)

        self.concatenated = np.concatenate(self.input(), axis=self.axis)
        return self.concatenated

    def _split_indices(self) -> List[int]:
        shapes = self.input_shape()

        axis = self.axis

        running_sum = shapes[0][axis]

        indices = [running_sum]

        for shape in shapes[1:-1]:
            dim = shape[axis]
            running_sum += dim
            indices.append(running_sum)

        return indices

    def backprop_inputs(self, grad: np.ndarray, *args, **kwargs) -> Tuple[np.ndarray]:
        indices = self._split_indices()

        return tuple(np.split(grad, indices_or_sections=indices, axis=self.axis))
