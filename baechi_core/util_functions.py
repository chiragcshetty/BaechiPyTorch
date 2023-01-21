import torch
import config
import ctypes, gc
import psutil, os
from datetime import datetime

############################################# Print functions ###########################################
def print_exp_settings(args, exp_type ="baechi"):
    '''
    Print settings of the experiment being run
    input: experiment settings passed as arguments to the main program
    '''
    print("Model: ", args.model_name)
    print("Batch size: ", args.batch_size)
    if exp_type == "baechi":
        print("Baechi Scheme: ", args.sch)
        print("memory factor: " + str(args.mem_factor) + " (mem per device =" 
                    + humanize_num_bytes(config.MAX_AVAILABLE_MEMORY*args.mem_factor) + ")")
        print("no of gpus: "+ str(args.gpu_num))
        print("assigner type: " + args.assigner_type + "\n")

def print_gpu_memory():
    '''
    print memory of all available GPU's
    '''
    for i in range(torch.cuda.device_count()):
        #print(torch.cuda.get_device_name(i))
        print("GPU:", i)
        print('Memory Usage:')
        print('Allocated:', round(torch.cuda.memory_allocated(i)/1024**3,8), 'GB')
        print('Cached:   ', round(torch.cuda.memory_reserved(i)/1024**3,8), 'GB')
        #print("-----------------")
        #GPUtil.showUtilization()
        print("-----------")

def print_assigned_graph(return_graph):
    """
    helper function to print where each layer is assigned to
    :param return_graph: assigned DiGraph
    """
    my_record = {}
    print("Name : permanent_memory : weight : gpu_assigned: topo_order")
    for node in return_graph.nodes(data=True):
        my_record[node[1]['id']] = (node[1]['name'], node[1]['permanent_mem'], node[1]['weight'], node[1]['p'], node[1]['topo_order'])

    for i in range(len(return_graph.nodes)):
        print(i, my_record[i])

######################################### Graph enumerations ##################################################

# Get the leaf operations in a model. model.modules() gives not just the leaves, but higher levels as well
# Ref: https://stackoverflow.com/questions/54846905/pytorch-get-all-layers-of-model
# More explanation: https://discuss.pytorch.org/t/module-children-vs-module-modules/4551/4
def get_children(model: torch.nn.Module):
    # get children form model
    children = list(model.children())
    flatt_children = {}
    if children == []:
        # if model has no children; model is last child! :O
        return {id(model): model}
    else:
       # look for children from children... to the last child!
       for child in children:
            try:
                flatt_children.update(get_children(child))
            except TypeError:
                flatt_children.update(get_children(child))
    return flatt_children

## Get list of all layers in the model
def get_module_list(model):
    module_list = []
    def get_modules(module):
        sub_modules = module.__dict__['_modules']
        if len(sub_modules) and (module.__class__.__name__ not in basic_blocks):
            for name, sub_module in sub_modules.items():
                get_modules(sub_module)
        elif (args.model_name  == 'gnmt' and\
                sub_module.__class__.__name__ =='Dropout'):#To exclude dropout layers (they need not be moved to gpus)
            pass
        else:
            module_list.append(module)
    get_modules(model)
    return module_list


######################################## Memory Tracking functions ###########################################

## bytes to GB
def b2gb(x): return round(x/2**30,8)

def print_mem(gpu_id, cached=0, unit='B'):
    '''
    return/print memory of a given GPU.
    input:  gpu_id (0 to max gpu count-1)
            cached: flag inidcates what infto print
    '''
    if unit=='GB':
        mem_allocated = round(torch.cuda.memory_allocated(gpu_id)/1024**3,8)
        mem_cached    = round(torch.cuda.memory_reserved(gpu_id)/1024**3,8)
    else:
        mem_allocated = torch.cuda.memory_allocated(gpu_id)
        mem_cached    = torch.cuda.memory_reserved(gpu_id)
        
    if cached>0:
        print('Allocated:', mem_allocated , 'GB')
    if cached>1:
        print('Cached:   ', mem_cached    , 'GB')
    return mem_allocated, mem_cached


## Estimate size of the model (in GB or MB or B)
## i.e sum of sizes of all parameters of the model
def estimate_model_size(model, unit='B', to_print = True): 
    persistent_memory = 0
    for name, param in model.named_parameters():
        persistent_memory += param.element_size() * param.nelement()
    if unit == 'GB':
        gb_mem = round(persistent_memory/1024**3,8)
        if to_print:
            print("Estimated Model Memory:",gb_mem, "GB")
        return gb_mem
    elif unit == 'B':
        gb_mem = persistent_memory
        if to_print:
            print("Estimated Model Memory:",gb_mem, "Bytes")
        return gb_mem
    else:
        mb_mem = round(persistent_memory/1024**2,8)
        if to_print:
            print("Estimated Model Memory:", mb_mem, "MB")
        return mb_mem


## Compute memory oof a tuple - used for gnmt
def recursively_compute_tuple_mem(tup): 
    net_mem = 0
    if isinstance(tup, tuple):
        for ele in tup:
            if isinstance(ele, torch.Tensor):
                net_mem += float(torch.prod(torch.tensor(ele.size())))
            elif type(ele) is torch.nn.utils.rnn.PackedSequence: 
                net_mem += float(torch.prod(torch.tensor(ele.data.size())))
                net_mem += float(torch.prod(torch.tensor(ele.batch_sizes.size())))
            elif isinstance(ele, tuple):
                net_mem += recursively_compute_tuple_mem(ele)
            elif ele is None:
                pass
            else:
                print(ele)
                raise ValueError("Something wrong here. This type is not handled yet:", type(ele))
    else:
        raise ValueError("Input is not a tuple")
    return net_mem


def estimate_tensor_size(inp, unit='B'):
    input_size = 0
    if isinstance(inp, torch.Tensor): 
        input_size += float(torch.prod(torch.tensor(inp.size())))
    elif isinstance(inp, list): 
        for sub_inp in inp:
            if isinstance(sub_inp, torch.Tensor): input_size += float(torch.prod(torch.tensor(sub_inp.size())))
    elif isinstance(inp, tuple): #for gnmt
        input_size += recursively_compute_tuple_mem(inp)
    elif inp is None:
        pass
    else:
        print(inp)
        raise ValueError("Something wrong here. Please handle this type:", type(inp))

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

## Context to track memory usage of a given GPU
class TorchTracemalloc():
    def __init__(self, gpu_id):
        self.gpu_id = gpu_id

    def __enter__(self):
        self.begin = torch.cuda.memory_allocated(self.gpu_id)
        torch.cuda.reset_max_memory_allocated(self.gpu_id) # reset the peak gauge to zero
        return self

    def __exit__(self, *exc):
        self.end  = torch.cuda.memory_allocated(self.gpu_id)
        self.peak = torch.cuda.max_memory_allocated(self.gpu_id)
        self.used   = (self.end-self.begin)
        self.peaked = (self.peak-self.begin)

## Context to track memory usage of all available GPUs
normalize_fact = 10**6
class TorchTracemalloc_new():
    def __init__(self, gpu, printout = False): #gpu is dummy variable, not used
        self.printout = printout
        self.gpu = gpu
        self.no_dev = torch.cuda.device_count()

    def __enter__(self):
        self.begin = [torch.cuda.memory_allocated(gpu_id) for gpu_id in range(self.no_dev)]
        for gpu_id in range(self.no_dev):
            torch.cuda.reset_max_memory_allocated(gpu_id) # reset the peak gauge to zero
        return self

    def __exit__(self, *exc):
        self.end  = [torch.cuda.memory_allocated(gpu_id) for gpu_id in range(self.no_dev)]
        self.peak = [torch.cuda.max_memory_allocated(gpu_id) for gpu_id in range(self.no_dev)]
        self.used   = [(self.end[gpu_id]-self.begin[gpu_id]) for gpu_id in range(self.no_dev)]
        self.peaked = [(self.peak[gpu_id]-self.begin[gpu_id]) for gpu_id in range(self.no_dev)]
        if self.printout:
            print("All printed values are in bytes normalized with the factor ",normalize_fact )
            if self.gpu<0:
                for gpu_id in range(self.no_dev): 
                    print("GPU:",gpu_id,"\t Begin, Peak, End:", self.begin[gpu_id]/normalize_fact, self.peak[gpu_id]/normalize_fact,self.end[gpu_id]/normalize_fact)
                    print("GPU:",gpu_id,"\t Used, Peak:", (self.used[gpu_id])/normalize_fact, (self.peaked[gpu_id])/normalize_fact)
                    print("-"*10)
            else:
                gpu_id = self.gpu
                print("GPU:",gpu_id,"\t Begin, Peak, End:", self.begin[gpu_id]/normalize_fact, self.peak[gpu_id]/normalize_fact,self.end[gpu_id]/normalize_fact)
                print("GPU:",gpu_id,"\t Used, Peak:", (self.used[gpu_id])/normalize_fact, (self.peaked[gpu_id])/normalize_fact)
                print("-"*10)
            print("*"*20)

class NullContextManager(object):
    def __init__(self, dummy_resource=None):
        self.dummy_resource = dummy_resource
    def __enter__(self):
        return self.dummy_resource
    def __exit__(self, *args):
        pass


def humanize_num_bytes(num_bytes):
    """Returns a number of bytes string."""
    suffixes = ['B', 'KB', 'MB', 'GB', 'TB', 'PB']
    i = 0
    while num_bytes >= 1024 and i < len(suffixes)-1:
        num_bytes /= 1024.
        i += 1
    number_str = ('%.2f' % num_bytes).rstrip('0').rstrip('.')
    return '%s%s' % (number_str, suffixes[i])


############################################## Baechi realted utils ################################################
def topological_sort(model):
    """
    this helper function helps to generate the execution order based on dependecies
    """
    record = set()
    while len(record) < len(model.sub_module_nodes):
        #print(len(record),len(model.sub_module_nodes) )
        root_helper = set(model.sub_module_nodes.keys()) - record
        reordered_root_helper = []
        for elem in model.submodule_order:
            if elem in root_helper:
                reordered_root_helper.append(elem)
        reordered_root_helper += list(root_helper - set(reordered_root_helper))
        for elem in root_helper:
            parents = model.sub_module_nodes[elem].parent
            if parents is None or len(parents - record) == 0:
                model.sub_module_nodes[elem].id = len(record)
                record.add(elem)


def copy_p(assigned_graph, model):
    """
    helper function to add .p field based on the assigned DiGraph 
    """
    for node_id in model.sub_module_nodes:
        model.sub_module_nodes[node_id].p = assigned_graph.nodes[model.sub_module_nodes[node_id].id]["p"]
        model.sub_module_nodes[node_id].topo_order = assigned_graph.nodes[model.sub_module_nodes[node_id].id]["topo_order"]

     
def recursively_assign(inp, dev):
    """
    helper function to assign input recursively to a gpu Device
    """
    result = None
    if isinstance(inp, list):
        result = []
        for elem in inp:
            result.append(recursively_assign(elem, dev))
    else:
        if inp.device.index != dev:
            result = inp.cuda(dev)
        else:
            result = inp
    return result


def get_optimizer(optimizer_name, params, lr = None):

    if optimizer_name == 'adadelta':
        if lr == None:
            optimizer = torch.optim.Adadelta(params)
        else:
            optimizer = torch.optim.Adadelta(params, lr = lr)

    elif optimizer_name == 'adagrad':
        if lr == None:
            optimizer = torch.optim.Adagrad(params)
        else:
            optimizer = torch.optim.Adagrad(params, lr =lr)

    elif optimizer_name == 'adam':
        if lr == None:
            optimizer = torch.optim.Adam(params)
        else:
            optimizer = torch.optim.Adam(params, lr =lr)

    elif optimizer_name == 'rmsprop':
        if lr == None:
            optimizer = torch.optim.RMSprop(params)
        else:
            optimizer = torch.optim.RMSprop(params, lr =lr)

    elif optimizer_name == 'sgd':
        if lr == None:
            raise ValueError("Learning rate must be specified for SGD")
        else:
            optimizer = torch.optim.SGD(params, lr)
    #elif optimizer_name == <new_optimizer>: # add new
    #    optimizer = .....
    else:
        raise ValueError(
            'Optimizer [%s] was not recognized' % optimizer_name)
    return optimizer


def get_criterion(criterion_name):

    if criterion_name == 'mse':
        criterion = torch.nn.MSELoss()
    elif criterion_name == 'cross_entropy':
        criterion = torch.nn.CrossEntropyLoss()
    else:
        raise ValueError(
            'Criterion [%s] was not recognized' % criterion_name)
    return criterion

######### Memory clearing utils (especially useful in jupyter notebook env, to clean mem after runs) #####

### From https://discuss.pytorch.org/t/how-pytorch-releases-variable-garbage/7277
def memReport():
    for obj in gc.get_objects():
        if torch.is_tensor(obj):
            print(type(obj), obj.size())
    
def delTensors():
    for obj in gc.get_objects():
        if torch.is_tensor(obj):
            #print("Deleting: ", type(obj), obj.size())
            del obj
    
def cpuStats():
        print(sys.version)
        print(psutil.cpu_percent())
        print(psutil.virtual_memory())  # physical memory usage
        pid = os.getpid()
        py = psutil.Process(pid)
        memoryUse = py.memory_info()[0] / 2. ** 30  # memory use in GB
        print('memory GB:', memoryUse)

################################# Result recording utils #######################################################
## Record the results of a run in a file
def record_result(exp_type, args, gpu_time, used_mem_list=[], peaked_mem_list=[]):
    now = datetime.now()
    if exp_type == 'baechi':
        folder_name = "results/" + args.model_name + "_" + str(args.batch_size) + "_" + str(args.perturb_factor)
        if not os.path.isdir( folder_name ):
            os.mkdir(folder_name)
    else:
        folder_name = "results"
    
    if exp_type == 'baechi':
        f = open(folder_name + "/" +exp_type + "_" + args.model_name + "_" + args.sch + ".txt", "a")
    else:
        f = open(folder_name + "/" + exp_type + "_" + args.model_name + ".txt", "a")
    f.write("Date-Time:"+ str(now )+ "\n")
    f.write("batch_size = " +  str(args.batch_size) + "\n")
    if exp_type == 'baechi':
        f.write("memory factor = " + str(args.mem_factor) + "( mem per device =" 
                    + humanize_num_bytes(config.MAX_AVAILABLE_MEMORY*args.mem_factor) + ")"+ "\n")
        f.write("comm_factor = " + str(config.COMM_FACTOR)+ "\n")
        f.write("perturb factor = " + str(args.perturb_factor) +  "\n")
        f.write("no of gpus = "+ str(args.gpu_num)+ "\n")
        f.write("assigner type = " + args.assigner_type + "\n")
    f.write("STEP TIME(ms) = "+ str(gpu_time)+ "\n")
    if args.capture_info == 'capture_memory':
        print("-"*30)
        f.write("Memory per usuage device:"+ "\n")
        for i in range(len(used_mem_list)):
            f.write("\t GPU-" + str(i)+ ": used= "+ humanize_num_bytes(used_mem_list[i])+ ", peaked= "
                + humanize_num_bytes(peaked_mem_list[i])+ "\n" )
    f.write("-*"*50  + "\n")
    f.close()

    f = open("results/step_times.txt", "a")
    if os.path.getsize("results/step_times.txt") == 0:
        if exp_type == 'baechi':
            f.write(args.sch+":"+str(args.perturb_factor) + "\n")
        else:
            f.write("single_gpu"+ "\n")
    f.write(str(gpu_time) + "\n")
    f.close()


