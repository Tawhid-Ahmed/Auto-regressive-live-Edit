"""Test script to check if we can load BLIP2 model"""
import torch
import sys
import os

# Add current directory to path
sys.path.insert(0, os.path.dirname(__file__))

print("=" * 60)
print("Testing BLIP2 Model Loading")
print("=" * 60)

# Check GPU
if torch.cuda.is_available():
    print(f"CUDA Available: {torch.cuda.get_device_name(0)}")
    print(f"GPU Memory - Total: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB")
    print(f"GPU Memory - Allocated: {torch.cuda.memory_allocated(0) / 1024**3:.2f} GB")
    print(f"GPU Memory - Reserved: {torch.cuda.memory_reserved(0) / 1024**3:.2f} GB")
else:
    print("CUDA NOT Available!")
    sys.exit(1)

print("\nAttempting to load BLIP2 model...")
try:
    from utils import load_vllm_for_edit
    from utils.GLOBAL import model_path_map
    
    model_path = model_path_map['blip2-opt-2.7b']
    print(f"Model path: {model_path}")
    
    if not os.path.exists(model_path):
        print(f"ERROR: Model path does not exist: {model_path}")
        sys.exit(1)
    
    print("Loading model on cuda:0...")
    vllm = load_vllm_for_edit('blip2-opt-2.7b', 'cuda:0')
    print("SUCCESS: Model loaded!")
    
    print(f"\nGPU Memory After Load:")
    print(f"  Allocated: {torch.cuda.memory_allocated(0) / 1024**3:.2f} GB")
    print(f"  Reserved: {torch.cuda.memory_reserved(0) / 1024**3:.2f} GB")
    
except Exception as e:
    print(f"\nERROR: Failed to load model: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

print("\n" + "=" * 60)
print("Test completed successfully!")
print("=" * 60)
