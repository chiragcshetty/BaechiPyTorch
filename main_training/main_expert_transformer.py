import sys, os
sys.path.append("..")

import torch
import time
import networkx as nx
from torch import optim, nn
import numpy as np
import argparse

from torch.profiler import profile, record_function, ProfilerActivity
import gc

import config
import baechi_core.util_functions as utilsf
from model_library.model_nursery import get_model 

config.MAX_DEVICE_COUNT = torch.cuda.device_count()
parser = argparse.ArgumentParser(description='Expert placement for transformer training')

## Training setup
parser.add_argument('-b', '--batch-size', default=config.BATCH_SIZE , type=int, metavar='N', 
                        help='mini-batch size (default: 64)')

parser.add_argument('-n', '--num-run', default=config.NUM_RUN, type=int, 
                         help='how many training steps to run in the experiment')

parser.add_argument('-Ci', '--capture-info', default=config.CAPTURE_INFO, type=str, choices = ['capture_profile', 'capture_memory', None],
                         help='what info must be recorded during the training runs: timing profile (using torch profiler), \
                         or memory usuage or none.')

def main():
    args = parser.parse_args()

    print("Expert GPU run started. For transformer")
    model_name = "transformer"
    args.model_name = model_name

    utilsf.print_exp_settings(args, 'single_gpu')

    ################################################################################################################
    torch.cuda.synchronize(0);torch.cuda.synchronize(1);torch.cuda.synchronize(2);torch.cuda.synchronize(3)
    torch.cuda.set_per_process_memory_fraction(1.0, 0)
    torch.cuda.set_per_process_memory_fraction(1.0, 1)
    torch.cuda.set_per_process_memory_fraction(1.0, 2)
    torch.cuda.set_per_process_memory_fraction(1.0, 3)

    ################################################################################################################
    ## Setup the model
    model, inp_size_single, opt_size, model_info, lr, inpt_factor = get_model(\
                                                                        model_name, repetable=0, with_gpu_split=True)

    Nrun = args.num_run

    use_optimizer = (len(list(model.parameters()))>0)
    ################################################################################################################
    # Setup the inputs
    print("Setting up inputs for transformer")

    torch.random.manual_seed(0)
    inp_data = [None for _ in range(len(inp_size_single))]
    for i in range(len(inp_size_single)): # important to have requires_grad=True, else grad_fn may be None (in some cases)
        if inp_size_single[i][0] == "int":
            inp_data[i] = torch.randint(model_info['vocab_size'], (Nrun, args.batch_size,) + inp_size_single[i][1])
        else:
            inp_data[i] = torch.randn((Nrun, args.batch_size,) + inp_size_single[i])
    labels_data = torch.randn( (Nrun, args.batch_size,) + opt_size)

    if use_optimizer:# ^~
        optimizer = optim.SGD(model.parameters(), lr = 0.0001); optimizer.zero_grad()
    criterion = nn.MSELoss()

    context = utilsf.NullContextManager() #change
    if args.capture_info == 'capture_profile':
        context = profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA])
    elif args.capture_info == 'capture_memory':
        context = utilsf.TorchTracemalloc_new(-1,False)
    ################################################################################################################
    ## training
    print("Strating training.")
    
    out_sum = 0  # sum of all elements of output -> For correctness check
    times = []
    times_f_single =[] #forward runtime
    times_b_single =[] #backward runtime

    for run_no in range(Nrun):
        print("Run number:", run_no)

        with context as ctx:
            start = time.time()
            #--------- move input and labels -----------

            inp = [None for _ in range(len(inp_size_single))]
            for i in range(len(inp_size_single)):
                inp[i] = inp_data[i][run_no]
                if inp_size_single[i][0] != "int":
                    inp[i].requires_grad = True
            labels = labels_data[run_no]
            labels = labels.to(1)
                
            if use_optimizer:
                optimizer.zero_grad()

            #--------------- forward ------------------
            f_start = time.time() 
            
            inp = tuple(inp)
            output = model(*inp)
            if isinstance(output, tuple): 
                output = output[0]

            torch.cuda.synchronize(0);torch.cuda.synchronize(1);torch.cuda.synchronize(2);torch.cuda.synchronize(3)
            #--------------- backward ------------------
            b_start = time.time()
            loss = criterion(output, labels)
            loss.backward(loss)
            if use_optimizer:
                optimizer.step()
            torch.cuda.synchronize(0);torch.cuda.synchronize(1);torch.cuda.synchronize(2);torch.cuda.synchronize(3)
            end = time.time()
            
        out_sum = out_sum + torch.sum(output.detach().clone())# for correctness check
        #print(out_sum) # for correctness check
        times.append(1000*(end-start))
        times_f_single.append(1000*(b_start-f_start))
        times_b_single.append(1000*(end-b_start))
        print("-*"*40)

    if args.capture_info == 'capture_profile':
        ctx.export_chrome_trace("traces/trace_expert.json")
    if args.capture_info == 'capture_memory':
        used_mem_list = ctx.end
        peaked_mem_list = ctx.peak


    single_gpu_time = np.mean(times[int(Nrun/4):])
    single_gpu_time_f = np.mean(times_f_single[int(Nrun/4):])
    single_gpu_time_b = np.mean(times_b_single[int(Nrun/4):])
    print("Mean time taken:", single_gpu_time)
    print("Mean forward time taken:", single_gpu_time_f)
    print("Mean backward time taken:", single_gpu_time_b)
    print()
    print("Output_sum:", out_sum)

        
    utilsf.record_result('singlegpu', args, single_gpu_time, used_mem_list, peaked_mem_list)


    #print_gpu_memory()
    del model
    try:
        del inp
    except:
        del inp_enc, inp_dec
        
    del output
    try:
        del labels
        del optimizer
        del loss
    except: pass
    gc.collect()
    torch.cuda.empty_cache()
    #print_gpu_memory()

if __name__ == '__main__':
    main()