import torch


if not torch.cuda.is_available():
    raise RuntimeError("torch cannot see a CUDA device")

device_name = torch.cuda.get_device_name(0)
x = torch.randn(1024, 1024, device="cuda")
y = x @ x

print("hello from lupine remote runner")
print(f"torch version: {torch.__version__}")
print(f"cuda devices visible: {torch.cuda.device_count()}")
print(f"cuda device 0: {device_name}")
print(f"matmul checksum: {float(y.sum()):.4f}")
