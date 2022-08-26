import torch
import time
import numpy as np

print("GPU speed test running....")
torch.cuda.synchronize(0);torch.cuda.synchronize(1);torch.cuda.synchronize(2);torch.cuda.synchronize(3)
matrix = torch.randn((5000,5000))
times = {}

## Runs dummy computes on each gpu and records the time to complete
## Baechi assumes that all devices have the same compute power/speed

for dev in range(torch.cuda.device_count()):
    print("Device:", dev)
    matrix = matrix.to(dev)
    torch.cuda.synchronize(dev)
    times[dev] = []
    
    for _ in range(8):
        start = time.time()
        for _ in range(20):
            matrix = torch.matmul(matrix, matrix)
        torch.cuda.synchronize(dev)
        end = time.time()
        times[dev].append(1000*(end-start))
    time.sleep(2)
    
    print("Time(ms):", np.mean(times[dev][4:])); print("*"*20)

print()
print("Note: For NVIDIA GeForce RTX 2080, the ideal value is ~ 500ms")