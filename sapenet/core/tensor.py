import numpy as np
from typing import Optional, Sequence
from enum import Enum

class TensorContext:
    def __init__(self, is_constant: bool = False, left: Optional[Tensor] = None, right: Optional[Tensor] = None, operation: Optional[str] = None):
        if not is_constant and not all((left, right)): raise ValueError("Left and right tensor must be set in non-constant contexts.")

        self.is_constant = is_constant

        self.left = left
        self.right = right
        self.operation = operation

    @staticmethod
    def constant():
        return TensorContext(is_constant=True, left=None, right=None)

class Tensor:
    FLOAT = np.float32
    INT = np.int32

    def __init__(self, data: Optional[np.ndarray] = None):
        if isinstance(data, np.ndarray) and data.dtype not in (Tensor.FLOAT, Tensor.INT): data = Tensor.FLOAT(data)

        self.data = data
        self.context = TensorContext.constant()

        self._projected_size = -1

    @property
    def size(self):
        return self.data.size if isinstance(self.data, np.ndarray) else self._projected_size

    @staticmethod
    def random_tensor(size: int|Sequence[int]) -> Tensor:
        return Tensor(data=np.random.random(size=size))

    def __add__(self, other): return _tensor_bin_op(a=self, b=other, operation='add')
    def __sub__(self, other): return _tensor_bin_op(a=self, b=other, operation='subtract')
    def __mul__(self, other): return _tensor_bin_op(a=self, b=other, operation='multiply')
    def __truediv__(self, other): return _tensor_bin_op(a=self, b=other, operation='divide')

def _tensor_bin_op(a: Tensor, b: Tensor, operation: str) -> Tensor:
    t = Tensor()

    t.context = TensorContext(is_constant=False, left=a, right=b, operation=operation)
    return t