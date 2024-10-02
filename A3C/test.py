import torch

if torch.cuda.is_available():
    device = torch.cuda.get_device_properties(0)
    print(f"Name: {device.name}")
    print(f"CUDA Cores: {device.multi_processor_count * 128}")  # 每个流式多处理器 (SM) 通常包含128个核心
else:
    print("No CUDA-enabled GPU available")
