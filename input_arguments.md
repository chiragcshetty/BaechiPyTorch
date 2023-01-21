Input argument details:

The three kinds of runs possible:
1. Run baechi on the model, followed by trainig oon the distributed model: main_baechi.py
2. Run training on a single device: main_single_gpu.py
3. (FFor transfomer) run the training with expert placement: main_expert_transformer.py

Default arguments and other settings are listed in config.py
The settings can be changed through the command line options listed below.

#############################################################################################################
#####################################  For main_baechi.py  ##################################################
If no change is required just run:
python main_baechi.py 

*Results for the run will stored in the 'results' folder named as baechi_{model}_{sch}_{perturb_factor}
*Traces from the run (if 'capture_profile' flag is used) will be in the 'traces' folder


========================= Arguments ===================================

--run-type              =   type of run
                            defualt: config.RUNTYPE
                            choices: "training", "inference"

'-Ng', '--gpu-num'      =   Number of GPU to use for baechi
                            defualt: config.NUM_GPUS
                            choices: int, 1 to max no of gpu's available (i.e torch.cuda.device_count())

'-Pr', '--prof-rounds'  =   No. of rounds the profiler runs & averages the profiles
                            (inital 1/4th are ignored while averaging)
                            defualt: config.PROFILING_ROUNDS
                            choices: int>0

'-Pg', '--prof-gpu-id'  =   Which gpu to do the profiling on.
                            defualt: config.PROFILING_GPU
                            choices: 0 to num_gpus-1

'-sch', '--sch'         =   Baechi algorithm scheme to be used
                            defualt: config.SCHEME
                            choices: "sct", "etf" or "topo" (config.SCHEME_LIST)

'--topo-type'           =   If topo scheme is choosen, should a cap= (mem required)/(num devices) be applied?
                            defualt: config.TOPO_TYPE
                            choices: "no_cap", "with_cap"

'-At', '--assigner-type'=   Type of assigner to be used
                            -   Assigner moves the layers to devices according to the generated palcement and
                                wraps each layer with the communication protocol to pull inputs and send outputs
                                out of the device. Further, assigner can reorder the execution sequence of the 
                                layers based on the exact topo order that baechi used.
                            -   These can be done in multiple ways - hence the 5 choices for assigner type
                            defualt: config.ASSIGNER_TYPE
                            choices: see config.py for choices available

'-Pf', '--perturb-factor'=  Factor to perturb profiles (comm and compute times) by for sensitivity test. 
                            -   Profiles are perturbed by a uniform rv in range (1 +/- perturb-factor).
                            defualt: config.PERTURBATION_FACTOR
                            choices: positive float. Typically range to use -> 0 to 0.2

'-Mf', '--mem-factor'   =   Fraction of max_available_memory at each device allowed for placement
                            -   Used for incufficient memory experiments. 
                            defualt: config.MEM_FACTOR
                            choices: Typically 0.3 to 1.0. If more than 1.0, clipped to 1.0

-*-*-*-*-*-*-* Training setup -*-*-*-*-*-*

'-b', '--batch-size'    =   Mini batch size
                            defualt: config.BATCH_SIZE
                            choices: int - typically 32-64 for inception_v3

'-m', '--model-name'    =   Name of the model to be used
                            defualt: config.MODEL_NAME
                            choices: see config.MODEL_LIST

'-Mt', '--model-type'   =   Type of model (for internal use only)
                            - mainly used to determine the input/output sizes and formats 
                              (eg: packed tensors in gnmt etc)
                            - "inception" type can be used for most image models
                            defualt: "inception"
                            choices: currently "inception", "gnmt", "transformer"

'-rep', '--repetable'   =   Initiates with fixed parameters, and fixed inputs, lables
                            - This was experiemnts are repetable across runs
                            - Also used to check correctness of assingner 
                              (eg: identify sync bugs across devices)
                            defualt: config.REPETABLE
                            choices: 0, 1

'-n', '--num-run'       =   How many training steps to run in the experiment
                            defualt: config.NUM_RUN
                            choices: int>0

'-Ci', '--capture-info' =   What info must be recorded during the training runs: 
                            - Timing profile (using torch profiler) or Memory usuage or None 
                            - Note: using timing profiler may have an overhead and increase steptimes
                            defualt: config.CAPTURE_INFO
                            choices: "capture_profile", "capture_memory", None

'-Cc', '--correctness-check' =  Set it to 1 to check correctness of the assigners wrapper. It assigns random
                                device to each node. Set repetable also to 1 and compare the output generated 
                                by this run and from single_gpu_run.
                                default: 0
                                choices: 0,1

#################################################################################################################
############################################  For main_single_gpu.py ############################################
Runs the entire model on a single gpu 

If no change is required just run:
python main_single_gpu.py 

*Results for the run will stored in the 'results' folder named as singlegpu_{model}.txt

========================= Arguments ===================================

--run-type              =   type of run
                            defualt: config.RUNTYPE
                            choices: "training", "inference"
'-g', '--gpu-id'        =   Which gpu to run training on. Range = 
                            defualt: config.PROFILING_GPU
                            choices: 0 to num_gpus-1

-*-*-*-*-*-*-*-*-*-*-*-* Training setup -*-*-*-*-*-*-*-*-*-*-*-*
Same as main_baechi.py except for '--correctness-check'

################################################################################################################
#######################################  For main_expert_transformer.py  #######################################
Runs a simple manual placemment for transformer - with encoder on one device (gpu 0) and the decoder on another (gpu 1)

If no change is required, run:
python main_expert_transformer.py

*Results for the run will stored in the 'results' folder named as singlegpu_transformer.txt (yes, its a misnomer)

========================= Arguments ===================================
Same as "--batch-size", "--num-run", "--capture-info" from above