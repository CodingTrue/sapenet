import numpy as np

class Tensor:
    FLOAT = np.float32
    INT = np.int32

    def __init__(self, data: np.ndarray):
        if data.dtype not in (Tensor.FLOAT, Tensor.INT): raise ValueError(f"Datatype '{data.dtype}' is not supported.")

        self._data = data