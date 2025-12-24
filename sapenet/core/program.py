import pyopencl as cl
import numpy as np
from dataclasses import dataclass

from sapenet.utils import read_kernel
from .device import Device
from .tensor import Tensor
from .kernel import KernelRegistry

F_COPY_HOST_PTR = cl.mem_flags.COPY_HOST_PTR
F_RO = cl.mem_flags.READ_ONLY

@dataclass
class ComputeGraphEntry:
    function_name: str
    arguments: list[Tensor]
    output: Tensor

@dataclass
class DataBufferAttributes:
    buffer_name: str
    elements: int = 0
    offset: int = 0

@dataclass
class TensorAttributes:
    buffer: str
    buffer_index: int
    size: int
    offset: int

    def to_buffer_reference(self):
        return self.buffer

class Program:
    def __init__(self, compute_graph: list[ComputeGraphEntry]):
        self._compute_graph = compute_graph

        self._compute_kernel = None

        self._constant_tensors = []
        self._work_tensors = []
        
    def build(self):
        kernel_registry = KernelRegistry.instance()

        tensor_map = {}
        program_source = []
        kernel_calls = []
        dependencies = []

        constant_buffer_attributes = DataBufferAttributes(buffer_name="constant_data")
        work_buffer_attributes = DataBufferAttributes(buffer_name="work_data")

        for entry in self._compute_graph:
            current_kernel = kernel_registry.get_kernel(identifier=entry.function_name)
            current_kernel_source, current_kernel_identifier = current_kernel.get_kernel_variant(memory_regions=[tensor.context().is_constant() for tensor in entry.arguments])

            for tensor in (*entry.arguments, entry.output):
                if tensor in tensor_map: continue
                is_constant = tensor.context().is_constant()

                data_buffer_attributes = constant_buffer_attributes if is_constant else work_buffer_attributes
                size = current_kernel.get_buffer_output_size(arguments=entry.arguments) if tensor == entry.output else tensor.size()

                tensor_map[tensor] = TensorAttributes(
                    buffer=data_buffer_attributes.buffer_name,
                    buffer_index=data_buffer_attributes.elements,
                    size=size,
                    offset=data_buffer_attributes.offset,
                )
                tensor._projected_size = size

                data_buffer_attributes.offset += size
                data_buffer_attributes.elements += 1

                (self._constant_tensors if is_constant else self._work_tensors).append(tensor)

            kernel_call_arguments = (
                'G_ID',
                *(tensor_map[tensor].buffer for tensor in entry.arguments),
                tensor_map[entry.output].buffer,
                *(str(tensor_map[tensor].offset) for tensor in (*entry.arguments, entry.output)),
                str(tensor_map[entry.output].size)
            )
            kernel_call = f"{current_kernel_identifier}({', '.join(kernel_call_arguments)});"

            if not current_kernel_identifier in dependencies:
                program_source.append(current_kernel_source)
                dependencies.append(current_kernel_identifier)
            kernel_calls.append(kernel_call)

        program_source.append(
            read_kernel(path="base_kernel.cl").replace('$BODY_SECTION', '\n\t'.join(kernel_calls))
        )
        source = '\n'.join(program_source)

        self._compute_kernel = cl.Program(
            Device.default()._context,
            source
        ).build().compute_kernel

        return self

    def run(self):
        if not self._compute_kernel: raise RuntimeError("Program needs to be built before running.")

        device = Device.default()
        ctx = device._context
        queue = device._queue

        constant_data = np.concatenate([tensor._data for tensor in self._constant_tensors])
        work_data = np.zeros(sum(tensor.size() for tensor in self._constant_tensors), dtype=Tensor.FLOAT)

        constant_data_buffer = cl.Buffer(context=ctx, flags=F_COPY_HOST_PTR | F_RO, hostbuf=constant_data)
        work_data_buffer = cl.Buffer(context=ctx, flags=F_COPY_HOST_PTR, hostbuf=work_data)

        self._compute_kernel.set_args(constant_data_buffer, work_data_buffer)

        cl.enqueue_nd_range_kernel(queue=queue, kernel=self._compute_kernel, global_work_size=(max(tensor.size() for tensor in self._work_tensors), 1), local_work_size=None)
        cl.enqueue_copy(queue, work_data, work_data_buffer)

        pointer = 0
        for tensor in self._work_tensors:
            size = tensor.size()
            tensor._data = work_data[pointer:pointer + size]

            pointer += size

    @staticmethod
    def evaluate_tensor(tensor: Tensor):
        if tensor.context().is_constant(): return tensor.data

        compute_graph = []
        _build_compute_graph(tensor=tensor, compute_graph=compute_graph)

        Program(compute_graph=compute_graph).build().run()

        return tensor._data

def _build_compute_graph(tensor: Tensor, compute_graph: list[ComputeGraphEntry]):
    ctx = tensor.context()
    if ctx.is_constant(): return

    _build_compute_graph(tensor=ctx._left, compute_graph=compute_graph)
    _build_compute_graph(tensor=ctx._right, compute_graph=compute_graph)

    compute_graph.append(ComputeGraphEntry(
        function_name=ctx._operation,
        arguments=(ctx._left, ctx._right),
        output=tensor
    ))