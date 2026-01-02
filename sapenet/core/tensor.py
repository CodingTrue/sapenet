import numpy as np
from typing import Optional, Sequence
from enum import Enum

class TensorContext:
    def __init__(self, is_constant: bool = False, children: Optional[Sequence[Tensor]] = [], operation: Optional[str] = None):
        if not is_constant and not children: raise ValueError("Tensor children must be set in non-constant contexts.")

        self.is_constant = is_constant

        self.children = children
        self.operation = operation

    @staticmethod
    def constant():
        return TensorContext(is_constant=True, children=[])

class Tensor:
    FLOAT = np.float32
    INT = np.int32

    def __init__(self, data: Optional[np.ndarray] = None):
        if isinstance(data, np.ndarray) and data.dtype not in (Tensor.FLOAT, Tensor.INT): data = Tensor.FLOAT(data)

        self.data = data
        self.context = TensorContext.constant()

        self._projected_size = -1
        self._projected_shape = (-1, -1)

    @property
    def size(self):
        return self.data.size if isinstance(self.data, np.ndarray) else self._projected_size

    @property
    def shape(self):
        return self.data.shape if isinstance(self.data, np.ndarray) else self._projected_shape

    @staticmethod
    def random_tensor(size: int|Sequence[int]) -> Tensor:
        return Tensor(data=np.random.random(size=size))

    def __add__(self, other): return _tensor_operation(children=(self, other), operation='add')
    def __sub__(self, other): return _tensor_operation(children=(self, other), operation='subtract')
    def __mul__(self, other): return _tensor_operation(children=(self, other), operation='multiply')
    def __truediv__(self, other): return _tensor_operation(children=(self, other), operation='divide')

    def __neg__(self):
        if self.context.is_constant: return Tensor(data=-self.data)

        return _tensor_operation(children=(self,), operation="negate")

def _tensor_operation(children: Sequence[Tensor], operation: str) -> Tensor:
    t = Tensor()

    t.context = TensorContext(is_constant=False, children=children, operation=operation)
    return t