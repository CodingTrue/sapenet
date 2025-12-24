import numpy as np
from sapenet.core import Device, Tensor, Program

# Select avaiable OpenCL device
# device_type = Device.GPU | Device.CPU | Device.ALL
cl_device = Device.devices(device_type=Device.ALL)[0]
Device.default(cl_device)

print(f"Selected device: {cl_device.name} | {Device.default()}")
print("-"*32)

# Defining tensors
a = Tensor(np.array([1, 2, 3, 4, 5, 6], dtype=Tensor.FLOAT))
b = Tensor(np.array([6, 5, 4, 3, 2, 1], dtype=Tensor.FLOAT))

# Element-wise operations
c = a + b
d = b + c

e = c * d

# Evaluate top level tensor
e_value = Program.evaluate_tensor(tensor=e) # returns value of evaluated tensor

print(f"{c.data = }") # [7.  7.  7.  7.   7.  7.]
print(f"{d.data = }") # [13. 12. 11. 10.  9.  8.]
print(f"{e.data = }") # [91. 84. 77. 70. 63. 56.]
print(f"{e_value = }") # [91. 84. 77. 70. 63. 56.]