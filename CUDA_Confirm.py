import torch

print("CUDA Available:", torch.cuda.is_available())
print("Current Device:", torch.cuda.current_device())
print("Device Count:", torch.cuda.device_count())
print("Device Name:", torch.cuda.get_device_name(0))
