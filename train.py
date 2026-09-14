import ctypes


def check(code, call):
    if code != 0:
        raise RuntimeError(f"{call} failed with CUDA code {code}")


cuda = ctypes.CDLL("libcuda.so.1")

check(cuda.cuInit(0), "cuInit")

count = ctypes.c_int()
check(cuda.cuDeviceGetCount(ctypes.byref(count)), "cuDeviceGetCount")
if count.value < 1:
    raise RuntimeError("no CUDA devices visible")

device = ctypes.c_int()
check(cuda.cuDeviceGet(ctypes.byref(device), 0), "cuDeviceGet")

name = ctypes.create_string_buffer(256)
check(cuda.cuDeviceGetName(name, len(name), device), "cuDeviceGetName")

print("hello from lupine remote runner")
print(f"cuda devices visible: {count.value}")
print(f"cuda device 0: {name.value.decode()}")
