import pyopencl as cl
import numpy as np
from dataclasses import dataclass

from sapenet.utils import read_kernel
from .device import Device
from .tensor import Tensor
from .kernel import KernelRegistry

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
    is_constant: bool

    def to_buffer_reference(self):
        return self.buffer

class Program:
    def __init__(self, compute_graph: list[ComputeGraphEntry]):
        self._compute_graph = compute_graph

        self._program = None
        self._tensor_map = {}
        
    def build(self):
        kernel_registry = KernelRegistry.instance()

        injected_dependencies = []
        kernel_source = []
        body_source = []

        constant_buffer_attributes = DataBufferAttributes(buffer_name="constant_data")
        work_buffer_attributes = DataBufferAttributes(buffer_name="work_data")

        for entry in self._compute_graph:
            current_kernel = kernel_registry.get_kernel(identifier=entry.function_name)
            current_kernel_source, current_kernel_identifier = current_kernel.get_kernel_variant(memory_regions=[tensor.context().is_constant for tensor in entry.arguments])

            for tensor in (*entry.arguments, entry.output):
                if tensor in self._tensor_map: continue
                is_constant = tensor.context().is_constant

                data_buffer_attributes = constant_buffer_attributes if is_constant else work_buffer_attributes
                size = current_kernel.get_buffer_output_size(arguments=entry.arguments) if tensor == entry.output else tensor.size()

                self._tensor_map[tensor] = TensorAttributes(
                    buffer=data_buffer_attributes.buffer_name,
                    buffer_index=data_buffer_attributes.elements,
                    size=size,
                    offset=data_buffer_attributes.offset,
                    is_constant=is_constant
                )
                tensor._projected_size = size

                data_buffer_attributes.offset += size
                data_buffer_attributes.elements += 1

            kernel_call_arguments = (
                'G_ID',
                *(self._tensor_map[tensor].buffer for tensor in entry.arguments),
                self._tensor_map[entry.output].buffer,
                *(str(self._tensor_map[tensor].offset) for tensor in (*entry.arguments, entry.output)),
                str(self._tensor_map[entry.output].size)
            )
            kernel_call = f"{current_kernel_identifier}({', '.join(kernel_call_arguments)});"

            if not current_kernel_identifier in injected_dependencies:
                kernel_source.append(current_kernel_source)
                injected_dependencies.append(current_kernel_identifier)
            body_source.append(kernel_call)

        kernel_source.append(
            read_kernel(path="base_kernel.cl").replace('$BODY_SECTION', '\n\t'.join(body_source))
        )
        source = '\n'.join(kernel_source)

        self._program = cl.Program(
            Device.default()._context,
            source
        ).build()

        return self

    def run(self):
        if not self._program: raise RuntimeError("Program needs to be builded before running.")

        constant_tensors, work_tensors = [], []
        for tensor in self._tensor_map:
            (constant_tensors if self._tensor_map[tensor].is_constant else work_tensors).append(tensor)

        constant_data = np.concatenate([tensor._data for tensor in constant_tensors])
        work_data = np.zeros(sum(tensor.size() for tensor in work_tensors), dtype=Tensor.FLOAT)

        _f_copy_host_ptr = cl.mem_flags.COPY_HOST_PTR
        _f_ro = cl.mem_flags.READ_ONLY

        device = Device.default()
        ctx = device._context
        queue = device._queue
        compute_kernel = self._program.compute_kernel

        constant_data_buffer = cl.Buffer(context=ctx, flags=_f_copy_host_ptr | _f_ro, hostbuf=constant_data)
        work_data_buffer = cl.Buffer(context=ctx, flags=_f_copy_host_ptr, hostbuf=work_data)

        compute_kernel.set_args(constant_data_buffer, work_data_buffer)
        cl.enqueue_nd_range_kernel(queue=queue, kernel=compute_kernel, global_work_size=(max(tensor.size() for tensor in work_tensors), 1), local_work_size=None)

        cl.enqueue_copy(queue, work_data, work_data_buffer)

        pointer = 0
        for tensor in work_tensors:
            size = tensor.size()
            tensor._data = work_data[pointer:pointer + size]

            pointer += size

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
        function_name=ctx._operation,
        arguments=(ctx._left, ctx._right),
        output=tensor
    ))