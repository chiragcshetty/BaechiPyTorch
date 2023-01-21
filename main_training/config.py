RUNTYPE                 = 'training'  # 'training'/'inference'
MAX_DEVICE_COUNT        = 4

#------ Baechi setup -----------
PROFILING_ROUNDS        = 20          # help: 'rounds for profiler'
NUM_GPUS                = 4           # help: 'number of gpu to use. Must be less than torch.cuda.device_count() '
PROFILING_GPU           = 2           # help: 'which gpu to place the profiler. Range = (0 to NUM_GPUS-1)'
SCHEME                  = 'etf'       # help: algorithm scheme to be used - sct, etf, topo
SCHEME_LIST             = ['etf', 'sct', 'topo']
TOPO_TYPE               = 'with_cap'              # Choices: ['no_cap', 'with_cap'], If topo scheme is choosen, 
                                                  # should a cap= (mem required)/(num devices) be applied?
                                                  # All Baechi topo experiments were 'with_cap'
# Maximum memory (RAM) available for placement at each device after subtracting memory occupied 
# by the library files. We use 8GB devices. Upto ~1GB maybe occupied by these libraries - but it
# is not a fixed value. Sometimes Baechi may generate a placement, but actual execution may OOM.
# In that case reduce max_available_memory to upto (8000000000 - 1.5*1080000000) and try again.
# Set between (8000000000 - 1.5*1080000000) to (8000000000 - 0.5*1080000000)
MAX_AVAILABLE_MEMORY    = (8000000000 - 1080000000) 


#------ Model setup -----------
MODEL_LIST     = ["inception_v3", "transformer", "gnmt", "LinearModel", "ShortLinear", "ParallelThreeLayer", 
                    "inception_dummy", "InceptionE", "OneLayer", "ShortInceptionE", "InceptionE2"]

MODEL_NAME              = 'inception_v3' 
BATCH_SIZE              = '64'          # help: 'batch_size'

# Some models can be initiated with fixed parameters, so that experiments are repetable
REPETABLE               = 0             # 0 or 1
OPTIMIZER_DEFAULT       = 'sgd'  
CRITERION_DEFAULT       = 'mse'  

#------ Tunings for experiments -----------
PERTURBATION_FACTOR     = 0.0      # factor to perturb profiles by for sensitivity test. 
                                   # Profiles is perturbed by a uniform rv in (1 +/- PERTURBATION_FACTOR)
NODE_WEIGHT_FACTOR      = 1.0      # Tune node weights i.e node execution time
COMM_FACTOR             = 1.2      # Tune edge weights i.e communication time between nodes
MEM_FACTOR              = 1        # Fraction of 'max_available_memory' at each device allowed for placement
                                   # - used for incufficient memory experiments 

SLOPE_COMM_TIME         = 1.788e-07   # slope of comm_time_vs_data_size (experiment in comm_time_calibration.ipynb)
INTERCEPT_COMM_TIME     = 0.1         # intercept of the same (ms) 


#-----------------------------------------------
# Names of blocks in the model that will be treated as a single node when building graph (Sec X.X in paper)
# basic blocks should not be very large, else will lead to huge memory overestimation by the profiler 
BASIC_BLOCKS = ["BasicConv2d", "ScaledDotProductAttention", "PositionalEncoding", "OneHead","MultiHeadAttention", "PositionwiseFeedForward"] #blocks which are not broken down further and placed on one GPU
OPTIMIZERS    = ['sgd', 'adadelta', 'adagrad', 'adam', 'rmsprop']
CRITERIONS    = ['mse', 'cross_entropy']

# - If REVERSE_GRAPH_FLAG true, backward graph is used instead of forward
# - Node weights can be increased using NODE_WEIGHT_FACTOR. Backward runtime 
#   of a node is typically 2-3 times that of forward.
# - We do no precisely measure the backward runtime of modules, since that
#   requires a backward_pre_hook which PyTorch does not provide natively. (Following 
#   patch can add that functionality, but we don't do it
#   : https://github.com/msr-fiddle/pipedream/tree/pipedream/profiler/torchmodules/torchprofiler )
REVERSE_GRAPH_FLAG = False 
if REVERSE_GRAPH_FLAG:
    NODE_WEIGHT_FACTOR = 2.0
    COMM_FACTOR = 2*COMM_FACTOR

NUM_RUN  = 12   # how many training steps to run in the experiment

# what must be recorded during the training runs: timing profile (using torch profiler), or memory usuage
# or none. (previously called 'CONTEXT_TYPE')
#  
CAPTURE_INFO = 'capture_memory' #' capture_profile', 'capture_memory', None


#'gpu_topo_based', 'global_topo_based',   ----using both stream and reordering 
# 'only_stream_no_reordering',            ----streams only
# 'no_stream_no_reordering',              ----plain
# 'no_stream_only_reordering' ]           ----reordering only (using threads)
# 'gnmt_assigner_plain'                   ----for gnmt
# 'gnmt_assigner_stream'                  ----for gnmt
ASSIGNER_LIST = ['gpu_topo_based', 'global_topo_based', 'only_stream_no_reordering', 'no_stream_only_reordering',
                    'no_stream_no_reordering', 'gnmt_assigner_plain', 'gnmt_assigner_stream']
ASSIGNER_TYPE = 'gpu_topo_based'


#TODO: Baechi pytorch with GNMT has correctness and performance issues. So not allowing it to run
ALLOW_GNMT    = False



#----------------------------- Assumptions --------------------------------------
# In placed rx is assumed to begin with tx - but it is not fully true  
#---------------------------------------------------------------------------------