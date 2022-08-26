import sys, os
sys.path.append("..")

import argparse
import torch
from utils import logger
import time
from torch import optim, nn
import numpy as np

## This is a template for image models. 
## Steps:
##      1. add your model in model_library/models
##      2. add your model info in model_nursery.py (acc. to given template there)
##      3. pass the model_name as a command line argument
##      4. criterion and optimizer can also be specified as command line args
##      5. by default training happens on dummy random data. To use your own data,
##         scroll down to the "setup the inputs" section and add 'inp_data' and 'labels_data'
##         of the mentioned size 

## Typical runs:
## python main_baechi_template.py (runs default settings in config.py)
#-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!!-!-!-!-!-! Baechi Imports -!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!!-!-!-!-!-!
import config
from model_library.model_nursery import get_model
import baechi_core.baechi_profiler as profiler 
import baechi_core.baechi_graph as baechi_graph
import baechi_core.util_functions as utilsf

from baechi_core.placer.m_etf import m_etf
from baechi_core.placer.m_sct import m_sct
from baechi_core.placer.m_topo import m_topo
import baechi_core.assigners as assigners 
config.MAX_DEVICE_COUNT = torch.cuda.device_count()
#-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!!-!-!-

_LOGGER = logger.get_logger(__file__, level=logger.INFO)

#---------------------------------------- Process Input Arguments ------------------------------------------
parser = argparse.ArgumentParser(description='Baechi PyTorch Training')

## Experiment Setup
parser.add_argument('--run-type', default=config.RUNTYPE , type=str, help='Type of run: training or inference')

parser.add_argument('-Ng', '--gpu-num', default=config.NUM_GPUS, type=int, choices=range(1,config.MAX_DEVICE_COUNT+1), 
                        metavar='N', help='Number of GPU to use. Max value is torch.cuda.device_count()')

## Profiler Settings
parser.add_argument('-Pr', '--prof-rounds', default=config.PROFILING_ROUNDS, type=int,
                        metavar='P', help='no. of rounds the profiler runs & averages the profiles (default: 20)')

parser.add_argument('-Pg', '--prof-gpu-id', default=config.PROFILING_GPU, type=int, choices=range(0,config.MAX_DEVICE_COUNT), 
                        metavar='N', help='which gpu to do the profiling on. Range = (0 to NUM_GPUS-1)')

## Baechi Settings
parser.add_argument('-sch', '--sch', default=config.SCHEME, type=str, choices=config.SCHEME_LIST, 
                        help='baechi algorithm scheme to be used - sct, etf or topo')

parser.add_argument('--topo-type', default=config.TOPO_TYPE, type=str, choices=['no_cap', 'with_cap'], 
                        help='If topo scheme is choosen, should a cap= (mem required)/(num devices) be applied?')

parser.add_argument('-At', '--assigner-type', default=config.ASSIGNER_TYPE, type=str, choices = config.ASSIGNER_LIST, 
                         help='Type of assigner to be used (see config.py for choices available)')

parser.add_argument('-Pf', '--perturb-factor', default=config.PERTURBATION_FACTOR, type=float, 
                         help='factor to perturb profiles by for sensitivity test. \
                                Profiles is perturbed by a uniform rv in (1 +/- PERTURBATION_FACTOR).\
                                Typically range to use -> 0 to 0.2')

parser.add_argument('-Mf', '--mem-factor', default=config.MEM_FACTOR , type=float,  
                         help='Fraction of max_available_memory at each device allowed for placement.\
                               Used for incufficient memory experiments. Typically 0.3 to 1.0 ')

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
                         help='how many training steps to run in the experiment (no of epochs)')
            
parser.add_argument('-op', '--optimizer-name', default=config.OPTIMIZER_DEFAULT, type=str, choices=config.OPTIMIZERS, 
                         help='Name of optimizer to use. Can add new optimizers in baechi_core/util_functions.py -> get_optimizer()\
                                    Also add to config.OPTIMIZERS')

parser.add_argument('-ct', '--criterion-name', default=config.CRITERION_DEFAULT, type=str, choices=config.CRITERIONS, 
                         help='Name of criterion to use. Can add new optimizers in baechi_core/util_functions.py -> get_criterion()\
                                    Also add to config.CRITERIONS')

parser.add_argument('-Ci', '--capture-info', default=config.CAPTURE_INFO, type=str, choices = ['capture_profile', 'capture_memory', None],
                         help='what info must be recorded during the training runs: timing profile (using torch profiler), \
                         or memory usuage or none.')

parser.add_argument('-Cc', '--correctness-check', default=0, type=int, choices=[0,1], 
                         help='Set it to 1 to check correctness of the assigners wrapper. It assigns random\
                          device to each node. Set repetable also to 1 and compare the output generated by this run\
                          and from single_gpu_run.')

#-----------------------------------------------------------------------------------------------------
def main():
    ##### parse arguments ####
    args = parser.parse_args()
    if args.mem_factor>1.0:
        _LOGGER.info("Given mem_factor value is {}. It has been capped at 1.0.".format(args.mem_factor))
        args.mem_factor = 1.0
    if args.mem_factor<=0.0:
        raise ValueError("mem_factor can't be negative")

    _LOGGER.info( "\n"+"*-"*30 +\
                "\nInput arguments:\n" +\
                ''.join(f'{k}=\t{v}\n' for k, v in vars(args).items()) + "\n" +\
                "*-"*30)


    #-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!!-!-!-!-!-! Baechi Setup -!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!!-!-!-!-!-!
    _LOGGER.info("Starting Baechi Runtime.")
    utilsf.print_exp_settings(args)

    for dev in range(args.gpu_num):
        _LOGGER.info("Initializing device {} ".format(dev)) 
        torch.cuda.synchronize(dev)
        torch.cuda.set_per_process_memory_fraction(1.0, dev)

    #-!-!-!-!-!-!-!-!-!-!-!-!-!-!- Details: model, criterion, optimizer -!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!

    _LOGGER.info("Getting the model {} from model nursery ".format(args.model_name))
    model, inp_size_single, opt_size, model_info, lr, inpt_factor = get_model(args.model_name, args.repetable)
    criterion = utilsf.get_criterion(args.criterion_name)

    ############################ Profiling #############################

    start_profile_time = time.time()
    profiled_model = profiler.Profiling(args.run_type, model, args.model_type, args.prof_gpu_id, args.prof_rounds,\
                                input_size = inp_size_single, output_size = opt_size, batch_size = args.batch_size,\
                                model_info = model_info)
    final_output = profiled_model.run(criterion = criterion,\
                                      optimizer_name = args.optimizer_name, learning_rate = lr)
    end_profile_time = time.time()
    _LOGGER.info("Profiling time =  {} sec".format((end_profile_time -start_profile_time) ))

    ###################### Build the models' Baechi graph ######################
    graph_info = {}
    graph_info['m_comm_time_vs_data_size'] = config.COMM_FACTOR*config.SLOPE_COMM_TIME 
    graph_info['b_comm_time_vs_data_size'] = config.INTERCEPT_COMM_TIME
    graph_info['perturb_factor']           = args.perturb_factor
    graph_info['node_weight_factor']       = config.NODE_WEIGHT_FACTOR 

    start_graph_time = time.time()
    return_graph, profiled_model = baechi_graph.build_graph(final_output, profiled_model, graph_info)
    utilsf.topological_sort(profiled_model)
    end_graph_time = time.time()
    _LOGGER.info("Graph creation time =  {} sec".format(end_graph_time -start_graph_time ))

    ###################### Build the device graph ############################

    device_list={} 
    for dev in range(config.MAX_DEVICE_COUNT):
        device_list[dev] = {'name': '/job:localhost/replica:0/task:0/device:XLA_GPU:'+str(dev),\
                            'memory_size': args.mem_factor*config.MAX_AVAILABLE_MEMORY , 'type': ''}
    available_devices     = range(args.gpu_num)
    available_device_list = {k:device_list[k] for k in available_devices}
    DEVICE_GRAPH_MULTIPLE = baechi_graph.create_device_graph(available_device_list)
    _LOGGER.info("Device graph created with {} gpus. Details:".format(args.gpu_num))
    for dev in available_device_list:
        _LOGGER.info("Device {}. Memory = {}".format(dev, 
                    utilsf.humanize_num_bytes(available_device_list[dev]['memory_size']) ))


    ############ Feed the model and device graphs to choosen placement Baechi algorithm ############# 

    placement_start_time = time.time()
    if args.sch == "sct":
        _LOGGER.info("Implementing m-SCT")
        placed_op_graph = m_sct(return_graph, DEVICE_GRAPH_MULTIPLE)
        
    elif args.sch == "etf":
        _LOGGER.info("Implementing m-ETF")
        placed_op_graph = m_etf(return_graph, DEVICE_GRAPH_MULTIPLE)
        
    elif args.sch == "topo":
        _LOGGER.info("Implementing m-TOPO")
        if args.topo_type =="no_cap":
            _LOGGER.info("Topo without cap.")
            placed_op_graph = m_topo(return_graph, DEVICE_GRAPH_MULTIPLE, uniform = False)
        else:
            _LOGGER.info("Topo with cap.")
            placed_op_graph = m_topo(return_graph, DEVICE_GRAPH_MULTIPLE, uniform = True)
    placement_end_time = time.time()

    _LOGGER.info("Placement time = {}".format(placement_end_time - placement_start_time) )

    utilsf.copy_p(return_graph, profiled_model)
    utilsf.print_assigned_graph(return_graph)


    ################################# For printing info about the generated palcement ##############################
    ## Also gets gpu's of first and final node - needed to move inputs and labels to appropriate devices 
    first_gpu = -1
    print_parent_info = False 
    print("Printing placed graph information:")
    for node_id in profiled_model.sub_module_nodes: 
        print("Module:", profiled_model.sub_module_nodes[node_id].module)
        print("id:", id(profiled_model.sub_module_nodes[node_id].module))
        curr_gpu_id = profiled_model.sub_module_nodes[node_id].p
        assert (curr_gpu_id < args.gpu_num) , "Invalid device allotted to a node" 
        print("Assigned GPU:", curr_gpu_id)
        print("Execution order:",profiled_model.sub_module_nodes[node_id].execution_order)
        print("Topo order:",profiled_model.sub_module_nodes[node_id].topo_order)
        if print_parent_info:
            print("Parent modules:")
            for par in profiled_model.sub_module_nodes[node_id].parent:
                print("\t", profiled_model.sub_module_nodes[par].module, id(profiled_model.sub_module_nodes[par].module) )
            print()
            print("Children modules:")
            for child in profiled_model.sub_module_nodes[node_id].children:
                print("\t", profiled_model.sub_module_nodes[child].module, id(profiled_model.sub_module_nodes[child].module) )
        if first_gpu < 0:
            first_gpu = curr_gpu_id
        print("-"*40)
        print()
    final_gpu = curr_gpu_id

    ##### Check topological order allotment is correct ####
    topo_list=[]
    for node_id in profiled_model.sub_module_nodes:
        topo_list.append(profiled_model.sub_module_nodes[node_id].topo_order)
        topo_list.sort()  
    for i in range(len(topo_list)-1):
        assert topo_list[i+1] == topo_list[i]+1, "Something wrong"

    ######################### Training/Infference according to  the generated palcement #########################################
    ##--*--*--*--*--*--*--*--*--*--*--*--*--*--*--*--*--*--*--*--*--*--*--*--*--*--*--*--*--*--*--*--*--*--*--*--*--*--*--*--*

    ## Set memory limits on each allowed device for this process
    for dev in range(args.gpu_num):
        torch.cuda.set_per_process_memory_fraction(args.mem_factor*1.0, dev)

    ################# Distribute model across devices acc to the generatted placement ###################
    assigner = assigners.assigner(profiled_model, args.assigner_type, args)
    for dev in range(args.gpu_num):
        torch.cuda.synchronize(dev) 
    ############################################## Setup the Inputs ####################################################

    _LOGGER.info("Setting up inputs:")

    #----------------------------------------------------------------------
    inp_size = (args.batch_size,) + inp_size_single
    opt_size_tup = (args.batch_size, opt_size)
    inp_data    = torch.randn((args.num_run,) + inp_size)*(inpt_factor)
    labels_data = torch.randn((args.num_run,) +opt_size_tup)*0.5
    #### OR configure your own data ##
    # inp_data    = entire input  set of size args.num_run(# of epoch) X args.batch_size X inp_size_single
    # labels_data = entire labels set of size args.num_run(# of epoch) X args.batch_size X opt_size
    #----------------------------------------------------------------------
        
    optimizer = utilsf.get_optimizer(args.optimizer_name, profiled_model.model.parameters(), lr = lr)

    ## Choose what info must be recorded during runs
    context = utilsf.NullContextManager() #change
    if args.capture_info == 'capture_profile':
        ## This may slow down the training steps
        from torch.profiler import profile, record_function, ProfilerActivity
        context = profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA])
    elif args.capture_info == 'capture_memory':
        context = utilsf.TorchTracemalloc_new(-1,False)

    ########################################### Training runs ################################################

    if args.run_type == "training":

        out_sum = 0

        times = []
        times_f =[]
        times_b =[]
        
        for run_no in range(args.num_run):
            print("Epoch:", run_no)
            with context as ctx:
                start = time.time()
                #--------- move input and labels -----------
                inp = inp_data[run_no].to(first_gpu)   #TODO: optimize by overlapping with computation
                labels = labels_data[run_no].to(final_gpu)
  
                optimizer.zero_grad()
                
                #--------------- forward ------------------
                f_start = time.time()
                output = profiled_model.model(inp)

                output = assigner.wait_for_threads(output)
                
                for dev in range(args.gpu_num):
                    torch.cuda.synchronize(dev)

                assigner.clear_records()
                    
                #--------------- backward ------------------
                b_start = time.time()
                loss = criterion(output, labels)
                loss.backward(loss)
                optimizer.step()

                for dev in range(args.gpu_num):
                    torch.cuda.synchronize(dev)
                end = time.time()

            times.append(1000*(end-start))
            times_f.append(1000*(b_start-f_start))
            times_b.append(1000*(end-b_start))
            print("-*"*40)

        if args.capture_info == 'capture_profile':
            ctx.export_chrome_trace("traces/trace_baechi.json")
        if args.capture_info == 'capture_memory':
            used_mem = ctx.end
            peaked_mem = ctx.peak

        gpu_time = np.mean(times[int(args.num_run/4):])
        gpu_time_f = np.mean(times_f[int(args.num_run/4):])
        gpu_time_b = np.mean(times_b[int(args.num_run/4):])
        print("Mean time taken:", gpu_time)
        print("Mean forward time taken:", gpu_time_f)
        print("Mean backward time taken:", gpu_time_b)
        print()

    if args.capture_info == 'capture_memory':
        utilsf.record_result('baechi', args, gpu_time, used_mem, peaked_mem)


if __name__ == '__main__':
    main()