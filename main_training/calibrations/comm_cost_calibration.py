import torch
import time
import numpy as np
#import matplotlib.pyplot as plt

## batches of np.rands of size (300,300) with given batch_sizes
## is transferred between devices N times and average transfer time 
## (ignoring first N/4 runs) is obtained

# units: milliseconds/byte

#------------------------------------------------------------------------------
def estimate_tensor_size(inp, unit='B'):
    input_size = 0
    if isinstance(inp, torch.Tensor): 
        input_size += float(torch.prod(torch.tensor(inp.size())))
    if isinstance(inp, list): 
        for sub_inp in inp:
            if isinstance(sub_inp, torch.Tensor): input_size += float(\
                                torch.prod(torch.tensor(sub_inp.size())))

    input_size = input_size*torch.rand((1,1)).element_size() # multiply by 4
    if unit == 'GB':
        gb_mem = round(input_size/1024**3,8)
        #print("Estimated Input/Output Memory:",gb_mem, "GB")
        return gb_mem
    if unit == 'B':
        gb_mem = input_size
        #print("Estimated Input/Output Memory:",gb_mem, "B")
        return gb_mem
    else:
        mb_mem = round(input_size/1024**2,8)
        #print("Estimated Input/Output Memory:", mb_mem, "MB")
        return mb_mem
#------------------------------------------------------------------------------

N = 8
#batch_size = [1,4,8,16,32,64,128,256,1024,2048,4096, 8192]
batch_size = [1,4,8,16,32,64,128,256]


gpu_ids = range(torch.cuda.device_count())
L = len(gpu_ids)
M = np.zeros((L,L)) #slopec
C = np.zeros((L,L)) #intercepts


for start_gpu  in gpu_ids:
    for end_gpu in gpu_ids:    
        d = {}
        print("Start GPU:", start_gpu, " | End GPU:", end_gpu)
        for n in batch_size:
            #dummy = torch.rand((n*2500,10, 64))
            print("Batch size:", n)

            times = []
            for i in range(N):
                dummy = torch.rand((n,300,300))
                dsize = estimate_tensor_size(dummy)
                dummy0 = dummy.to(start_gpu)
                torch.cuda.synchronize(start_gpu);torch.cuda.synchronize(end_gpu);

                start = time.time()
                dummy1 = dummy0.to(end_gpu)
                torch.cuda.synchronize(start_gpu);torch.cuda.synchronize(end_gpu);
                end = time.time()
                times.append(1000*(end-start))

            comm_time = np.mean(times[round(N/4):])

            del dummy1
            del dummy0
            torch.cuda.empty_cache()

            d[dsize] = comm_time

        lists = sorted(d.items()) # sorted by key, return a list of tuples
        x, y = zip(*lists)       # unpack a list of pairs into two tuples   
        z = np.polyfit(x, y, 1)
        M[start_gpu][end_gpu] = z[0]
        C[start_gpu][end_gpu] = z[1]
        print("m = ", z[0], " | c = ", z[1])
        print("-"*30)

# Compute average m and c across all device pairs

print()
print("Results:")

print();print("-"*30)
print("M:"); print()
for start_gpu  in gpu_ids:
    for end_gpu in gpu_ids:
        print("m for GPU-{} to GPU-{} = {}".format(start_gpu, end_gpu, M[start_gpu][end_gpu]))

print();print("-"*30)
print("C:"); print()
for start_gpu  in gpu_ids:
    for end_gpu in gpu_ids:
        print("c for GPU-{} to GPU-{} = {}".format(start_gpu, end_gpu, C[start_gpu][end_gpu]))

for gid in gpu_ids:
    assert M[gid][gid]<10**(-9), "Same device comm seems large for {} with m = {} ".format(\
                                    gid,M[gid][gid] )

m = np.sum(M)/(L*L - L) #only cross-device, contribution of same device is small
c = np.sum(C)/(L*L - L)

print()
print("*-"*30)
print("Update the following in main_training/config.py:")
print("SLOPE_COMM_TIME = ", m)
print("INTERCEPT_COMM_TIME = ", c)
print("*-"*30)

