import pyopencl as cl
import numpy as np
from dataclasses import dataclass, field
from typing import Sequence

from sapenet.utils import read_kernel
from .device import Device
from .tensor import Tensor
from .kernel import KernelRegistry, Kernel, KernelDimension

F_COPY_HOST_PTR = cl.mem_flags.COPY_HOST_PTR
F_RO = cl.mem_flags.READ_ONLY

BASE_KERNEL_SOURCE = read_kernel('base_kernel.cl')

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
class ComputeKernelSource:
    code_definitions: list[str] = field(default_factory=list)
    body_calls: list[str] = field(default_factory=list)
    constant_buffer_attributes: DataBufferAttributes = field(default_factory=lambda: DataBufferAttributes(buffer_name='constant_data'))
    work_buffer_attributes: DataBufferAttributes = field(default_factory=lambda: DataBufferAttributes(buffer_name='work_data'))
    max_work_size: tuple[int] = field(default_factory=tuple)

    def suggest_work_size(self, work_size: tuple[int]):
        self.max_work_size = work_size if work_size > self.max_work_size else self.max_work_size

    def get_source(self):
        return '\n'.join((
            '\n'.join(self.code_definitions),
            BASE_KERNEL_SOURCE.replace('$BODY_SECTION', '\n\t'.join(self.body_calls))
        ))

@dataclass
class ComputeKernelAttributes:
    kernel: cl.Kernel
    work_size: Sequence[int]

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
        self.compute_graph = compute_graph

        self.compute_kernels: list[ComputeKernelAttributes] = []

        self._tensor_map = {}
        self._constant_tensors = []
        self._work_tensors = []

    def build_partial_tensor_map(
            self,
            current_kernel: Kernel,
            constant_buffer_attributes: DataBufferAttributes,
            work_buffer_attributes: DataBufferAttributes,
            compute_graph_entry: ComputeGraphEntry
    ):
        for tensor in (*compute_graph_entry.arguments, compute_graph_entry.output):
            if tensor in self._tensor_map: continue
            is_constant = tensor.context.is_constant

            data_buffer_attributes = constant_buffer_attributes if is_constant else work_buffer_attributes
            size = current_kernel.get_buffer_output_size(arguments=compute_graph_entry.arguments) if tensor == compute_graph_entry.output else tensor.size

            self._tensor_map[tensor] = TensorAttributes(
                buffer=data_buffer_attributes.buffer_name,
                buffer_index=data_buffer_attributes.elements,
                size=size,
                offset=data_buffer_attributes.offset,
            )
            tensor._projected_size = size

            data_buffer_attributes.offset += size
            data_buffer_attributes.elements += 1

            (self._constant_tensors if is_constant else self._work_tensors).append(tensor)
        
    def build(self):
        kernel_registry = KernelRegistry.instance()
        compute_kernel_sources: list[ComputeKernelSource] = []
        last_kernel_dimension = None

        for entry in self.compute_graph:
            current_kernel = kernel_registry.get_kernel(identifier=entry.function_name)
            current_kernel_source, current_kernel_identifier = current_kernel.get_variant(memory_regions=[tensor.context.is_constant for tensor in entry.arguments])

            if not (last_kernel_dimension == current_kernel.dimension): compute_kernel_sources.append(ComputeKernelSource())
            cks = compute_kernel_sources[-1]

            self.build_partial_tensor_map(
                current_kernel=current_kernel,
                constant_buffer_attributes=cks.constant_buffer_attributes,
                work_buffer_attributes=cks.work_buffer_attributes,
                compute_graph_entry=entry,
            )

            call_arguments = current_kernel.get_call_arguments(
                arguments=entry.arguments,
                output=entry.output,
                tensor_map=self._tensor_map
            )

            cks.suggest_work_size(max([tensor.data.shape for tensor in entry.arguments]))
            cks.code_definitions.append(current_kernel_source)
            cks.body_calls.append(f"{current_kernel_identifier}({', '.join(call_arguments)});")

            last_kernel_dimension = current_kernel.dimension

        for cks in compute_kernel_sources:
            compute_kernel = cl.Program(
                Device.default().context,
                cks.get_source()
            ).build().compute_kernel

            self.compute_kernels.append(ComputeKernelAttributes(
                kernel=compute_kernel,
                work_size=cks.max_work_size
            ))

        return self

    def run(self):
        if not self.compute_kernels: raise RuntimeError("Program needs to be built before running.")

        device = Device.default()
        ctx = device.context
        queue = device.queue

        constant_data = np.concatenate([tensor.data.flatten() for tensor in self._constant_tensors])
        work_data = np.zeros(sum(tensor.size for tensor in self._work_tensors), dtype=Tensor.FLOAT)

        constant_data_buffer = cl.Buffer(context=ctx, flags=F_COPY_HOST_PTR | F_RO, hostbuf=constant_data)
        work_data_buffer = cl.Buffer(context=ctx, flags=F_COPY_HOST_PTR, hostbuf=work_data)

        events = []
        for ck in self.compute_kernels:
            ck.kernel.set_args(constant_data_buffer, work_data_buffer)

            wait_for = events[:]
            event = cl.enqueue_nd_range_kernel(
                queue=queue,
                kernel=ck.kernel,
                global_work_size=(max(tensor.size for tensor in self._work_tensors), 1),
                local_work_size=None,
                wait_for=wait_for
            )
            events.append(event)
        cl.enqueue_copy(queue, work_data, work_data_buffer)
        queue.finish()

        pointer = 0
        for tensor in self._work_tensors:
            size = tensor.size
            tensor.data = work_data[pointer:pointer + size]

            pointer += size

    @staticmethod
    def evaluate_tensor(tensor: Tensor):
        if tensor.context.is_constant: return tensor.data

        compute_graph = []
        found_tensors = []
        _build_compute_graph(tensor=tensor, compute_graph=compute_graph, found_tensors=found_tensors)

        Program(compute_graph=compute_graph).build().run()

        return tensor.data

def _build_compute_graph(tensor: Tensor, compute_graph: list[ComputeGraphEntry], found_tensors: list[Tensor]):
    ctx = tensor.context
    if ctx.is_constant or tensor in found_tensors: return

    for child in ctx.children:
        _build_compute_graph(tensor=child, compute_graph=compute_graph, found_tensors=found_tensors)

    compute_graph.append(ComputeGraphEntry(
        function_name=ctx.operation,
        arguments=ctx.children,
        output=tensor
    ))
    found_tensors.append(tensor)