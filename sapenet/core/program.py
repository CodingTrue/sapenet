import pyopencl as cl
import numpy as np
from dataclasses import dataclass
from importlib import resources

from .device import Device
from .tensor import Tensor
from sapenet import kernels

BUILTIN_DEPEDENCIES = (
    "builtin_lib.cl",
    "builtin_add.cl",
    "builtin_subtract.cl",
    "builtin_multiply.cl",
    "builtin_divide.cl",
    "base_kernel.cl"
)

def read_kernel(path: str) -> str:
    return (resources.files(kernels) / path).read_text()

@dataclass
class ComputeGraphEntry:
    function_name: str
    arguments: list[Tensor]
    output: Tensor

@dataclass
class TensorData:
    buffer: str
    buffer_index: int
    size: int
    offset: int
    is_constant: bool

    def to_buffer_reference(self):
        return f"{self.buffer}"

class Program:
    def __init__(self, compute_graph: list[ComputeGraphEntry]):
        self._compute_graph = compute_graph
        self._tensor_map = {}

        self._program = None

        self.make_tensor_map()

    def make_tensor_map(self):
        constant_elements, work_elements = 0, 0
        constant_offset, work_offset = 0, 0

        for entry in self._compute_graph:
            for tensor in entry.arguments + (entry.output,):
                if tensor in self._tensor_map: continue
                if tensor == entry.output:
                    tensor._data = np.zeros(entry.arguments[0].size(), dtype=Tensor.FLOAT)

                is_constant = tensor.context().is_constant

                buffer = 'constant_data' if is_constant else 'work_data'
                buffer_index = constant_offset if is_constant else work_offset
                size = tensor._data.size
                offset = constant_offset if is_constant else work_offset

                self._tensor_map[tensor] = TensorData(
                    buffer=buffer,
                    buffer_index=buffer_index,
                    size=size,
                    offset=offset,
                    is_constant=is_constant
                )

                if is_constant:
                    constant_elements += 1
                    constant_offset += size
                else:
                    work_elements += 1
                    work_offset += size

    def get_tensor_data(self, tensor: Tensor) -> TensorData:
        if not tensor in self._tensor_map: raise ValueError(f"Tensor '{tensor}' does not exist in tensor map.")
        return self._tensor_map[tensor]

    def build(self):
        body = []

        for entry in self._compute_graph:
            arguments = []
            argument_variant = ''
            offsets = []

            for tensor in entry.arguments + (entry.output,):
                tensor_buffer = self.get_tensor_data(tensor=tensor)

                arguments.append(tensor_buffer.to_buffer_reference())
                offsets.append(str(tensor_buffer.offset))

                if not tensor in entry.arguments: continue
                argument_variant += 'c' if tensor.context().is_constant else 'g'

            body.append(f"_{entry.function_name}_{argument_variant}(G_ID, {', '.join(arguments)}, {', '.join(offsets)}, {self.get_tensor_data(tensor=entry.arguments[0]).size});")
        source = '\n'.join(read_kernel(path=path) for path in BUILTIN_DEPEDENCIES).replace('$BODY_SECTION', '\n\t'.join(body))

        self._program = cl.Program(
            Device.default()._context,
            source
        ).build()

        return self

    def run(self):
        if not self._program: raise RuntimeError("Program needs to be builded before running.")

        constant_tensors, work_tensors = [], []
        for tensor, tensor_data in self._tensor_map.items():
            entry = (tensor, tensor_data.size)

            if tensor_data.is_constant: constant_tensors.append(entry)
            else: work_tensors.append(entry)

        constant_data = np.concatenate([tensor._data for tensor, _ in constant_tensors], dtype=Tensor.FLOAT)
        work_data = np.zeros(sum([size for _, size in work_tensors]), dtype=Tensor.FLOAT)

        min_elements = np.min((constant_data.shape, work_data.shape))

        constant_buffer = cl.Buffer(Device.default()._context, cl.mem_flags.READ_ONLY | cl.mem_flags.COPY_HOST_PTR, hostbuf=constant_data)
        work_buffer = cl.Buffer(Device.default()._context, cl.mem_flags.COPY_HOST_PTR, hostbuf=work_data)

        knl: cl.Kernel = self._program.compute_kernel
        knl.set_args(
            constant_buffer,
            work_buffer
        )

        queue = Device.default()._queue

        cl.enqueue_nd_range_kernel(queue, knl, (min_elements, 1), None).wait()
        cl.enqueue_copy(queue, work_data, work_buffer).wait()

        pointer = 0
        for tensor, size in work_tensors:
            tensor._data = work_data[pointer:pointer + size]
            pointer += size

        queue.finish()

    @staticmethod
    def evaluate_tensor(tensor: Tensor):
        if tensor.context().is_constant: return tensor._data

        compute_graph = []
        _build_compute_graph(tensor=tensor, compute_graph=compute_graph)

        Program(compute_graph=compute_graph).build().run()

        return tensor._data

def _build_compute_graph(tensor: Tensor, compute_graph: list[ComputeGraphEntry]):
    ctx = tensor.context()
    if ctx.is_constant: return

    _build_compute_graph(tensor=ctx._left, compute_graph=compute_graph)
    _build_compute_graph(tensor=ctx._right, compute_graph=compute_graph)

    compute_graph.append(ComputeGraphEntry(
        function_name=ctx._operation.name.lower(),
        arguments=(ctx._left, ctx._right),
        output=tensor
    ))