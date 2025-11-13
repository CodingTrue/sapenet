import numpy as np
from typing import Optional

class ComputeContext:
    def __init__(self, is_constant: bool = False, left: Optional[Tensor] = None, right: Optional[Tensor] = None):
        self._is_constant = is_constant

        if not is_constant and not all((left, right)): raise ValueError("Tensor left and tensor right must be set in non-constant contexts.")

        self._left = left
        self._right = right

    def is_constant(self) -> bool: return self._is_constant

    @staticmethod
    def constant():
        return ComputeContext(is_constant=True, left=None, right=None)

class Tensor:
    FLOAT = np.float32
    INT = np.int32

    def __init__(self, data: np.ndarray):
        if data.dtype not in (Tensor.FLOAT, Tensor.INT): raise ValueError(f"Datatype '{data.dtype}' is not supported.")

        self._data = data
        self._compute_context = ComputeContext.constant()

    def context(self) -> ComputeContext:
        return self._compute_context