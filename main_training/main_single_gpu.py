import sys, os
sys.path.append("..")

import torch
import time
import networkx as nx
from torch import optim, nn
import numpy as np
from utils import logger
import argparse

from torch.profiler import profile, record_function, ProfilerActivity
import gc

import baechi_core.util_functions as utilsf
from model_library.model_nursery import get_model
import config

_LOGGER = logger.get_logger(__file__, level=logger.INFO)
config.MAX_DEVICE_COUNT = torch.cuda.device_count()
parser = argparse.ArgumentParser(description='Single GPU PyTorch Training')

## Experiment Setup
parser.add_argument('--run_type', default=config.RUNTYPE , type=str, help='Type of run: training or inference')

parser.add_argument('-g', '--gpu-id', default=config.PROFILING_GPU, type=int, choices=range(0,config.MAX_DEVICE_COUNT), 
                        metavar='N', help='which gpu to run training on. Range = (0 to NUM_GPUS-1)')


## Training setup
parser.add_argument('-b', '--batch-size', default=config.BATCH_SIZE , type=int, metavar='N', 
                        help='mini-batch size (default: 64)')

parser.add_argument('-m', '--model-name', default=config.MODEL_NAME, type=str, choices=config.MODEL_LIST, 
                         help='name of the model to be used')

parser.add_argument('-Mt', '--model-type', default='inception', type=str, choices=['inception', 'gnmt', 'transformer' ], 
                         help='type of model (for internal use only)')
                         
parser.add_argument('-rep', '--repetable',default=config.REPETABLE , type=int, choices=[0,1],
                    help='Some models can be initiated with fixed parameters, so that experiments are repetable')

parser.add_argument('-n', '--num-run', default=config.NUM_RUN, type=int, 
                         help='how many training steps to run in the experiment')

parser.add_argument('-Ci', '--capture-info', default=config.CAPTURE_INFO, type=str, choices = ['capture_profile', 'capture_memory', None],
                         help='what info must be recorded during the training runs: timing profile (using torch profiler), \
                         or memory usuage or none.')

def main():
    args = parser.parse_args()

    #### TODO: Change if more types of 'transformer' or 'gnmt' models are added
    if args.model_name == 'gnmt':
        args.model_type = 'gnmt'
    elif args.model_name == 'transformer':
        args.model_type = 'transformer'
    else:
        args.model_type = 'inception'
    ####

    _LOGGER.info( "\n"+"*-"*30 +\
                "\nInput arguments:\n" +\
                ''.join(f'{k}=\t{v}\n' for k, v in vars(args).items()) + "\n" +\
                "*-"*30)
        

    _LOGGER.info("Starting Single GPU.")
    utilsf.print_exp_settings(args, "single_gpu")

    ################################################################################################################

    torch.cuda.synchronize(args.gpu_id)
    torch.cuda.set_per_process_memory_fraction(1.0, args.gpu_id)

    ################################################################################################################
    ## Setup the model
    model, inp_size_single, opt_size, model_info, lr, inpt_factor = get_model(args.model_name, args.repetable)

    with utilsf.TorchTracemalloc_new(-1,False) as tt0: 
        print("Model transferred to args.gpu_id: device", args.gpu_id)
        model = model.to(args.gpu_id)

    Nrun = args.num_run

    use_optimizer = (len(list(model.parameters()))>0)
    ################################################################################################################
    # Setup the inputs
    print("Setting up inputs....")
    if args.model_type  == 'gnmt':
        inp_enc_data = torch.randint(model_info['vocab_size'], (Nrun, args.batch_size, model_info['max_sequence_length']))
        inp_dec_data = torch.randint(model_info['vocab_size'], (Nrun, args.batch_size, model_info['max_sequence_length']))
        inp_seq_len_data = torch.sort(torch.randint(model_info['min_sequence_length'], model_info['max_sequence_length'],\
                                        (Nrun, args.batch_size)), descending=True)[0]
        labels_data = torch.empty(Nrun, args.batch_size, model_info['vocab_size'], dtype=torch.long).random_(2)
        if args.run_type == "training":
            optimizer = optim.SGD(model.parameters(), lr); 
            criterion = nn.CrossEntropyLoss(ignore_index=0, reduction='sum')
            
    elif args.model_type  == 'transformer': # only training
        assert (args.run_type == "training")
        torch.random.manual_seed(0)
        inp_data = [None for _ in range(len(inp_size_single))]
        for i in range(len(inp_size_single)): # important to have requires_grad=True, else grad_fn may be None (in some cases)
            if inp_size_single[i][0] == "int":
                inp_data[i] = torch.randint(model_info['vocab_size'], (Nrun, args.batch_size,) + inp_size_single[i][1])
            else:
                inp_data[i] = torch.randn((Nrun, args.batch_size,) + inp_size_single[i])
        labels_data = torch.randn( (Nrun, args.batch_size,) + opt_size)

        if use_optimizer:
            optimizer = optim.SGD(model.parameters(), lr = 0.0001); optimizer.zero_grad()
        criterion = nn.MSELoss()

    else:
        inp_size = (args.batch_size,) + inp_size_single
        opt_size_tup = (args.batch_size, opt_size)
        random_factor = [-2,-1,1,2]
        
        if args.repetable == 1:
            inp_data   = torch.ones((Nrun,) + inp_size)*(inpt_factor)
            labels_data = (torch.ones((Nrun,) +opt_size_tup)*0.5)
            #torch.random.manual_seed(0)
            #inp   = torch.randn((Nrun,) + inp_size)*(inpt_factor)
        else:
            inp_data   = torch.randn((Nrun,) + inp_size)*(inpt_factor)
            labels_data = (torch.randn((Nrun,) +opt_size_tup)*0.5)
            
        for run_no in range(Nrun): # just to make repetable input ones a little random
            inp_data[run_no]   = inp_data[run_no]*random_factor[run_no%4]

        if args.run_type == "training":
            optimizer = optim.SGD(model.parameters(), lr); 
            criterion = nn.MSELoss()

    context = utilsf.NullContextManager() #change
    if args.capture_info == 'capture_profile':
        context = profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA])
    elif args.capture_info == 'capture_memory':
        context = utilsf.TorchTracemalloc_new(-1,False)
    ################################################################################################################
    ## training
    print("Strating training....")
    if args.run_type == "training":    
        out_sum = 0  # sum of all elements of output -> For correctness check

        times = []
        times_f_single =[] #forward runtime
        times_b_single =[] #backward runtime

        for run_no in range(Nrun):
            print("Run number:", run_no);
            #torch.cuda.synchronize(0);torch.cuda.synchronize(1);torch.cuda.synchronize(2);torch.cuda.synchronize(3)
            
            with context as ctx:
                start = time.time()
                #--------- move input and labels -----------
                if args.model_type == "inception":
                    inp = inp_data[run_no]
                    inp = inp.to(args.gpu_id) 
                    labels = labels_data[run_no].to(args.gpu_id)

                elif args.model_type == "gnmt":
                    inp_enc = inp_enc_data[run_no]
                    inp_dec = inp_dec_data[run_no]
                    inp_seq_len = inp_seq_len_data[run_no]
                    labels = labels_data[run_no]

                    inp_enc = inp_enc.to(args.gpu_id)
                    inp_dec = inp_dec.to(args.gpu_id)
                    inp_seq_len = inp_seq_len.to(args.gpu_id)
                    labels = labels.to(args.gpu_id)

                elif args.model_type == "transformer":
                    inp = [None for _ in range(len(inp_size_single))]
                    for i in range(len(inp_size_single)):
                        inp[i] = inp_data[i][run_no]
                        if inp_size_single[i][0] != "int":
                            inp[i].requires_grad = True
                    labels = labels_data[run_no]
                    labels = labels.to(args.gpu_id)
                    
                if use_optimizer:
                    optimizer.zero_grad()

                #--------------- forward ------------------
                f_start = time.time() 
                if args.model_type == "inception":
                    output = model(inp)
                elif args.model_type == "gnmt":   
                    output = model(inp_enc, inp_seq_len, inp_dec)
                elif args.model_type == "transformer":
                    for i in range(len(inp_size_single)):
                        inp[i] = inp[i].to(args.gpu_id)
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
            ctx.export_chrome_trace("traces/trace_singlegpu.json")
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

    ################################################################################################################
    # inference runs
    if args.run_type == "inference":

        out_sum = 0
        times = []

        for run_no in range(Nrun):
            print("Run number:", run_no)
            #torch.cuda.synchronize(0);torch.cuda.synchronize(1);torch.cuda.synchronize(2);torch.cuda.synchronize(3)

            with context as ctx:
                if args.model_type == "inception":
                    inp = inp_data[run_no]
                    inp = inp.to(args.gpu_id)
                    #labels = labels_data[run_no].to(args.gpu_id)
                elif args.model_type == "gnmt":
                    inp_enc = inp_enc_data[run_no]
                    inp_dec = inp_dec_data[run_no]
                    inp_seq_len = inp_seq_len_data[run_no]
                    
                    inp_enc = inp_enc.to(args.gpu_id)
                    inp_dec = inp_dec.to(args.gpu_id)
                    inp_seq_len = inp_seq_len.to(args.gpu_id)
                    
                #labels = labels_data[run_no].to(args.gpu_id)
    
                start = time.time()
                with torch.no_grad():
                    if args.model_type == "inception":
                        output = model(inp)
                    elif args.model_type == "gnmt":   
                        output = model(inp_enc, inp_seq_len, inp_dec)
                torch.cuda.synchronize(0);torch.cuda.synchronize(1);torch.cuda.synchronize(2);torch.cuda.synchronize(3)
                end = time.time()
                #torch.cuda.empty_cache()
                times.append(1000*(end-start))
                out_sum = out_sum + torch.sum(output.detach().clone())

        if args.capture_info == 'capture_profile':
            ctx.export_chrome_trace("traces/trace_singlegpu.json")
        single_gpu_time = np.mean(times[int(Nrun/4):])
        print("Mean time taken:", single_gpu_time)
        print()
        output = out_sum
        
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