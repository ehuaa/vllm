import torch
def get_gpu_count():
    count = torch.cuda.device_count()
    return count

gpu_count = get_gpu_count()
print(gpu_count)
