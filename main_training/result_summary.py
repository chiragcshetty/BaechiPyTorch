import sys, os
sys.path.append("..")

import numpy as np
#from experiment_setting import *
import shutil
from datetime import datetime
import baechi_core.util_functions as utilsf

import argparse
import config

parser = argparse.ArgumentParser(description='Baechi PyTorch Training')

## Experiment Setup
parser.add_argument('-t','--exp-type', type=str, help='Type of experiment: baechi or single_gpu')

parser.add_argument('-ng', '--gpu-num', default=config.NUM_GPUS, type=int,\
                        metavar='N', help='Number of GPU to use. Max value is torch.cuda.device_count()')


## Baechi Settings
parser.add_argument('-sch', '--sch', default=config.SCHEME, type=str, choices=config.SCHEME_LIST, 
                        help='baechi algorithm scheme to be used - sct, etf or topo')

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

args = parser.parse_args()




result_file = "results/step_times.txt"
f = open(result_file, "r")
vals = f.readlines()
info = vals[0].split(":")
if args.exp_type == 'baechi':
        sch = info[0]
        perturb_factor = info[1][:-1]
else:
        sch = info[0][:-1]
vals = vals[1:]
step_times = [float(val) for val in vals]
avg_step_time = np.mean(step_times)
print("Average runtime is (in ms): ", avg_step_time ) 

if args.exp_type == 'baechi':
        folder_name = "results/"+args.model_name + "_" + str(args.batch_size) + "_" + str(args.perturb_factor)+"/"
else:
        folder_name = "results/"


now = datetime.now()
if avg_step_time>0:
        if args.exp_type == 'baechi':
                shutil.copyfile(result_file,folder_name+"step_times_" + args.exp_type
                        +"_"+args.model_name+"_"+str(args.batch_size)+'-'+args.sch+".txt")
        else:
                shutil.copyfile(result_file,folder_name+"step_times_" + args.exp_type
                        +"_"+args.model_name+"_"+str(args.batch_size)+".txt")
open(result_file, 'w').close() # clear the file

f = open("results/result_details.txt", "a")
f.write("Model: " + args.model_name + "\n")
f.write("Date-Time:" + str(now) + "\n")
f.write("batch_size = " +  str(args.batch_size) + "\n")
if args.exp_type == 'baechi':
        f.write("Baechi: "+ args.sch+ "\n")
        f.write("memory factor = " + str(args.mem_factor) + "( mem per device =" 
                + utilsf.humanize_num_bytes(config.MAX_AVAILABLE_MEMORY*args.mem_factor) + ")"+ "\n")
        f.write("comm_factor = " + str(config.COMM_FACTOR )+ "\n")
        f.write("perturb factor = " + str(args.perturb_factor) +  "\n")
        f.write("no of gpus = "+ str(args.gpu_num) + "\n")
        f.write("assigner type = " + args.assigner_type + "\n")
else:
        f.write("Single GPU run"+ "\n")
f.write("AVG STEP TIME(ms) = "+ str(avg_step_time)+ "\n")
f.write("-*"*50  + "\n")
f.close()