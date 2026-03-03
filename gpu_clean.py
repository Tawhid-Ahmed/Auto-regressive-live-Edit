import torch
import gc

# 1. Delete large tensors/models to remove references
# del model
# del optimizer
# del tensor_variable

# 2. Force Python garbage collection
gc.collect()

# 3. Release unused cached memory from PyTorch
torch.cuda.empty_cache()

print("GPU Memory Cache Cleared")
