import torch

print("CUDA available: ", torch.cuda.is_available())
print("cuDNN version: ", torch.backends.cudnn.version())
print("Available GPUs: ", torch.cuda.device_count())

