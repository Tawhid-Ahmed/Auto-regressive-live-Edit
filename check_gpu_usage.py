"""Quick script to check if models are on GPU and monitor memory usage"""
import torch

print("=" * 60)
print("GPU Status Check")
print("=" * 60)

if torch.cuda.is_available():
    print(f"✓ CUDA Available")
    print(f"  Device: {torch.cuda.get_device_name(0)}")
    print(f"  Total Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB")
    print(f"  Allocated: {torch.cuda.memory_allocated(0) / 1024**3:.2f} GB")
    print(f"  Reserved: {torch.cuda.memory_reserved(0) / 1024**3:.2f} GB")
    print(f"  Free: {(torch.cuda.get_device_properties(0).total_memory - torch.cuda.memory_reserved(0)) / 1024**3:.2f} GB")
    print()
    
    # Check if tensors are on GPU
    test_tensor = torch.randn(10, 10).cuda()
    print(f"✓ Test tensor on GPU: {test_tensor.device}")
    print(f"  Tensor location: {test_tensor.device.type}:{test_tensor.device.index}")
else:
    print("✗ CUDA NOT Available!")
    print("  Models will run on CPU (very slow!)")

print("=" * 60)
