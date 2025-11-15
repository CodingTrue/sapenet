import numpy as np
from typing import Optional
from enum import Enum

class OperationType(Enum):
    ADD         = 'add'
    SUBTRACT    = 'subtract'
    MULTIPLY    = 'multiply'
    DIVIDE      = 'divide'

class ComputeContext:
    def __init__(self, is_constant: bool = False, left: Optional[Tensor] = None, right: Optional[Tensor] = None, operation: Optional[OperationType] = None):
        self._is_constant = is_constant

        if not is_constant and not all((left, right)): raise ValueError("Tensor left and tensor right must be set in non-constant contexts.")

        self._left = left
        self._right = right
        self._operation = operation

    def is_constant(self) -> bool: return self._is_constant

    @staticmethod
    def constant():
        return ComputeContext(is_constant=True, left=None, right=None)

class Tensor:
    FLOAT = np.float32
    INT = np.int32

    def __init__(self, data: Optional[np.ndarray] = None):
        if isinstance(data, np.ndarray) and data.dtype not in (Tensor.FLOAT, Tensor.INT): raise ValueError(f"Datatype '{data.dtype}' is not supported.")

        self._data = data
        self._compute_context = ComputeContext.constant()

    def size(self):
        return self._data.size if isinstance(self._data, np.ndarray) else -1

    def context(self) -> ComputeContext:
        return self._compute_context

    def __add__(self, other): return _tensor_bin_op(a=self, b=other, operation=OperationType.ADD)
    def __sub__(self, other): return _tensor_bin_op(a=self, b=other, operation=OperationType.SUBTRACT)
    def __mul__(self, other): return _tensor_bin_op(a=self, b=other, operation=OperationType.MULTIPLY)
    def __truediv__(self, other): return _tensor_bin_op(a=self, b=other, operation=OperationType.DIVIDE)

def _tensor_bin_op(a: Tensor, b: Tensor, operation: OperationType) -> Tensor:
    t = Tensor()

    t._compute_context = ComputeContext(is_constant=False, left=a, right=b, operation=operation)

    return t