import pyopencl as cl
from typing import Optional

class Device:
    GPU = cl.device_type.GPU
    CPU = cl.device_type.CPU
    ALL = cl.device_type.ALL

    _default = None

    def __init__(self, cl_device: Optional[cl.Device] = None):
        if cl_device:
            self._device = cl_device
        else:
            cl_devices = Device.devices(device_type=Device.ALL)
            if len(cl_devices) == 0: raise RuntimeError("No OpenCL compatible devices found.")
            self._device = cl_devices[0]

        self._context = cl.Context(devices=[self._device])
        self._queue = cl.CommandQueue(self._context, device=self._device)

    @classmethod
    def default(cls, cl_device: Optional[cl.Device] = None) -> Device:
        if cl_device:
            cls._default = Device(cl_device=cl_device)

        if not cls._default:
            cls._default = Device()

        return cls._default

    @staticmethod
    def devices(device_type: cl.device_type = ALL) -> list[cl.Device]:
        cl_devices = []

        cl_platforms = cl.get_platforms()
        for cl_platform in cl_platforms:
            for cl_device in cl_platform.get_devices(device_type=device_type):
                cl_devices.append(cl_device)

        return cl_devices