from importlib import resources

from sapenet import kernels

def read_kernel(path: str) -> str:
    return (resources.files(kernels) / path).read_text()