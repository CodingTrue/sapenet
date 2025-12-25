import re
from typing import Sequence

from sapenet.core import Tensor
from sapenet.utils import read_kernel

GLOBAL_ID = 'G_ID'

class Kernel:
    def __init__(self, source_path: str, identifier: str):
        self.source = read_kernel(path=source_path)
        self.identifier = identifier

    def get_buffer_output_size(self, arguments: Sequence[Tensor]) -> int:
        return min([tensor.size for tensor in arguments])

    def get_kernel_variant(self, memory_regions: Sequence[bool]) -> tuple[str, str]:
        region_names = ['constant' if region else 'global' for region in memory_regions]
        region_alias = ''.join(name[0] for name in region_names)

        kernel_identifier = f'_{self.identifier}_{region_alias}'

        sub0 = re.sub(r'memory_region', lambda _: region_names.pop(0), string=self.source, count=len(region_names))
        sub1 = re.sub(f'{self.identifier}', kernel_identifier, string=sub0, count=1)

        return sub1, kernel_identifier

    def get_call_arguments(self, arguments: Sequence[Tensor], output: Tensor, tensor_map: dict[Tensor]) -> str:
        return (
            GLOBAL_ID,
            *(tensor_map[tensor].buffer for tensor in (*arguments, output)),
            *(str(tensor_map[tensor].offset) for tensor in (*arguments, output)),
            str(tensor_map[output].size)
        )

class KernelRegistry:
    _instance = None

    def __init__(self):
        self._entries = {}

    def register(self, kernel: Kernel) -> KernelRegistry:
        self._entries[kernel.identifier] = kernel
        return self

    def get_kernel(self, identifier: str) -> Kernel:
        if not identifier in self._entries: raise ValueError(f"Kernel '{identifier}' is not registered.")
        return self._entries[identifier]

    @classmethod
    def instance(cls):
        if not cls._instance:
            cls._instance = KernelRegistry()

        return cls._instance

_kernel_registry = KernelRegistry.instance()

_kernel_registry.register(kernel=Kernel(identifier='add', source_path="builtin_add.cl"))
_kernel_registry.register(kernel=Kernel(identifier='subtract', source_path="builtin_subtract.cl"))
_kernel_registry.register(kernel=Kernel(identifier='multiply', source_path="builtin_multiply.cl"))
_kernel_registry.register(kernel=Kernel(identifier='divide', source_path="builtin_divide.cl"))
_kernel_registry.register(kernel=Kernel(identifier='negate', source_path="builtin_negate.cl"))