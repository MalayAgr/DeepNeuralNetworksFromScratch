from __future__ import annotations

from typing import Union

from dnn.layers import BaseLayer
from dnn.input_layer import Input


LayerInput = Union[Input, BaseLayer]
