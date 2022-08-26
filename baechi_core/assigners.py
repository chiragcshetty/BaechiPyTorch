import torch
import time
import threading 

import util_functions as utilf
from utils import logger


_LOGGER = logger.get_logger(__file__, level=logger.INFO)

##TODO: parameters may be defined within blocks. Like "self.linear_att" in gnmt. They will be treated as lateral
# inputs. Anything better possible?

## update: two tensors on different device may have same id. so replaced id(output) with 'device_id' + '-' + id(output)

###############################################################
def recursively_move_to_gpu(tup, gpu):
    if isinstance(tup, tuple):
        tup_new = list(tup)
        for i, tu in enumerate(tup_new):
            if isinstance(tu, tuple):
                tu_new = recursively_move_to_gpu(tu, gpu)
                tup_new[i] = tu_new
            else:
                if torch.is_tensor(tu):
                    tup_new[i] = tu.to(gpu, non_blocking=True)
        tup_new = tuple(tup_new)
    else:
        print("input is not a tuple!")
    return tup_new

def get_first_element(t):
    if not isinstance(t, torch.Tensor):
        raise ValueError("Inputt must be tensor but got", type(t)) 
    first_ele = t[0]
    while len(first_ele.size())>0:
        first_ele = first_ele[0]
    return first_ele.data 

def recursively_get_output_ids(output_raw, module_gpu):
    output_dict = {}
    if isinstance(output_raw, tuple):
        for output in output_raw:
            if isinstance(output, torch.Tensor):
                output_id = str(module_gpu) +'-' +str(id(output))+'-'+ str(get_first_element(output))
                output_dict[output_id] = output
            elif isinstance(output, tuple):
                output_dict.update(recursively_get_output_ids(output, module_gpu))
            elif output is None:
                pass
            else:
                raise ValueError("This type not handled:", type(output))   
    else:
        print("input is not a tuple!")
    return output_dict
###############################################################

class Assign_gpu_topo(object):
    """
    Assigner that supports:
    - streams for cross-device communication
    - reordering of layer according to the topo order (using threads)
    - # of thread minimization using GPUwise topo order
    - clearing of intermediate outputs in this_assigner.own_outputs in inference runs
    # TODO: Relu and other inplace ops may cause correctness issue 
        when same tensor tensor is sent out parallely to another gpu
    """
    def __init__(self, model_wrapper, args):
        self.model = model_wrapper
        self.original_forwards = {}
        self.own_outputs = {} #just to keep reference to recent outputs
        self.output_record = {} # module_ouput_id: gpu_id -> (pointer to copy of the 'output' at 'gpu')
        self.thread_pool = [None for _ in range(len(self.model.sub_module_nodes))]
        self.gpuwise_execution_status = {i:[] for i in range(args.gpu_num)}
        self.gpuwise_queue = {i:[] for i in range(args.gpu_num)}
        self.execution_status = [None for _ in range(len(self.model.sub_module_nodes))]
        self.events = {i:[] for i in range(args.gpu_num)}
        self.wait_status = [True for _ in range(len(self.model.sub_module_nodes))]
        self.assigned = self.recur_move_layers_to_gpus()
        self.no_threads_launched = 0

        self.run_type   = args.run_type
        self.model_type = args.model_type

    
    def check_waiting(self, module_node):
        """
        returns True if the module_node's kernel needs to wait for some other node yet
        to be read by the cpu. If so, it must be launched as a thread and the main cpu 
        thread should proceed ahead. 'check_waiting' once false will be false thereafter 
        for the node.
        """
        _LOGGER.debug("Checking waiting status for node {}".format(module_node.name))
        this_assigner = self

        myTopo = module_node.gpu_topo_order
        myGlobalTopo = module_node.topo_order
        myGPU = module_node.p

        if this_assigner.wait_status[myGlobalTopo]==False: #cache of wait status. An optimization.
            return False
        
        ## A topologically precendent node hasn't been launched yet, so wait
        if sum(this_assigner.gpuwise_execution_status[myGPU][:myTopo])<myTopo:  # this is necessary
            _LOGGER.debug("A topologically precendent node hasn't been launched yet, so wait." )
            return True

        ## Check gpu-wise precedant's wait status
        if myTopo>0:
            if this_assigner.check_waiting(this_assigner.gpuwise_queue[myGPU][myTopo-1]):
                _LOGGER.debug("A gpuwise precedant is waiting, so wait." )
                return True  

        #Check if any of the parent is waiting
        for par_id in module_node.parent:
            par_node =  this_assigner.model.sub_module_nodes[par_id]
            if this_assigner.check_waiting(par_node): #parent is waiting
                _LOGGER.debug("Parent node {} is waiting on gpu {}. So wait".format(par_node.name,
                                 par_node.p) )
                return True

        this_assigner.wait_status[myGlobalTopo]= False #will be False henceforth
        _LOGGER.debug("Node need not wait. Safe to launch it in the main thread itself." )
        return False
 
    ## Function that wraps the forward function of the module_node
    def get_modified_forward(self, module_node, module_gpu, myTopo, myGlobalTopo):
        this_assigner = self

        module = module_node.module
        module_id = id(module)
        #--------------------------------------------------------------------------------------------
        ## Forward function wrapper
        def modified_forward(self, *inputs):
            result =[None]  # results are passed across threads as this mutable list

            ############## Real forward function launched as a thread or on main thread ##############
            def myForward(*inputs ):       
                ### Wait for nodes with topological precedence on the current GPU to be launched
                if myTopo>0:
                    prev_event = this_assigner.events[module_gpu][myTopo-1]
                    prev_event.wait()
                #_LOGGER.debug("Launching:", module_node.module )
                my_event = this_assigner.events[module_gpu][myTopo]
                #-------------------------------------------------------
                ### Wait for the rx streams from parent nodes to current gpu
                for par_id in module_node.parent:
                    par_node = this_assigner.model.sub_module_nodes[par_id]
                    par_gpu    = par_node.p
                    if par_gpu != module_gpu:
                        parent_event = this_assigner.events[par_gpu][par_node.gpu_topo_order]
                        parent_event.wait() ### Wait for parent of current node to be launched
                        par_stream = par_node.output_stream[module_gpu]['rx']
                        (torch.cuda.default_stream(device=module_gpu)).wait_stream(par_stream)
                #-------------------------------------------------------
                input_list = list(inputs)
                for i, inp in enumerate(input_list):
                    if myGlobalTopo>0 and isinstance(inp, list): 
                        inp = inp[0] #to extract value from the list passed by previous thread
                    is_lateral_input = 0
                    if torch.is_tensor(inp): #if input is tensor, get its copy on ther current GPU
                        inp_device = inp.get_device()
                        inp_id = str(inp_device) + '-' + str(id(inp))
                        if inp_device != module_gpu:
                            if inp_id in this_assigner.output_record:
                                ## get the input's copy on current gpu
                                if module_gpu in this_assigner.output_record[inp_id]:
                                    input_list[i] = this_assigner.output_record[inp_id][module_gpu]
                                elif "lateral" in this_assigner.output_record[inp_id]: 
                                    #input is in some GPU but not on this one
                                    is_lateral_input = 1
                                else:
                                    raise Exception("Invalid input recieved at module:", module_node.module)
                            else:
                                is_lateral_input = 1
                            if is_lateral_input: 
                                # Lateral inputs (input to model or masks in transformer), 
                                # which are not output of some previous layer
                                #assert (inp.get_device()<0) #Lateral inputs come from cpu
                                inp_transferred = inp.to(module_gpu)
                                if inp_id not in this_assigner.output_record:
                                    this_assigner.output_record[inp_id] = {}

                                #to prevent multiple copies (like masks in transformer)
                                this_assigner.output_record[inp_id][module_gpu] = inp_transferred 
                                input_list[i] = inp_transferred
                                #just to indicate this is a lateral input
                                this_assigner.output_record[inp_id]["lateral"] = None 
                        else:
                            input_list[i] = inp
                    else:
                        _LOGGER.debug("Input is not a tensor at {}, topo {}, inp {} ".format(module_node.name, 
                                    myGlobalTopo, inp) )
                inputs = tuple(input_list)
                #-------------------------------------------------------
                output = this_assigner.original_forwards[module_id](*inputs) 
                result[0] = output
                output_id = str(module_gpu) + '-' + str(id(output))
                ## for current output record pointers to its copies on child gpus
                this_assigner.output_record[output_id] = {} 
                        
                if this_assigner.run_type == "inference":                   ## just to keep a reference to thhe output
                   this_assigner.own_outputs[module_id]=output ## leads to correctness issue without this
                                                               ## where the memory (thus object ids) are reused
                ## Make output tx Copy Streams wait for the Compute Stream
                for child_id in module_node.children:
                    child_gpu = this_assigner.model.sub_module_nodes[child_id].p
                    if child_gpu != module_gpu:
                        this_assigner.output_record[output_id][child_gpu] = None
                        module_node.output_stream[child_gpu]['tx'].wait_stream(torch.cuda.default_stream(device=module_gpu))
                
                ## Start transferring the output to child gpus 
                one_of_child_gpu = module_gpu
                for child_id in module_node.children:
                    child_gpu = this_assigner.model.sub_module_nodes[child_id].p
                    if child_gpu != module_gpu:
                        one_of_child_gpu = child_gpu
                        if this_assigner.output_record[output_id][child_gpu] is None: 
                            #else output has already been transferred to child gpu
                            with torch.cuda.stream(module_node.output_stream[child_gpu]['tx']):
                                with torch.cuda.stream(module_node.output_stream[child_gpu]['rx']):
                                    this_assigner.output_record[output_id][child_gpu] = output.to(child_gpu)

                if this_assigner.run_type == "inference": #to not bloat memory
                    if one_of_child_gpu != module_gpu:
                        with torch.cuda.stream(module_node.output_stream[one_of_child_gpu]['tx']):
                            del this_assigner.own_outputs[module_id]
                my_event.set()
                return
            #----------------------------------------------------------------------------------------------------
            _LOGGER.debug("Module {}, GPU {}, global topo {}, gpuwise topo ".format( 
                module_node.name,module_node.p, myGlobalTopo, myTopo) )
            # wait as a thread if either:
            # a. any of the parent node is waiting OR
            # b. any of the nodes with topological precedence on current gpu have not been launched
            if this_assigner.check_waiting(module_node): #need to spawn a thread
                _LOGGER.debug("Launching as a thread")
                th = threading.Thread(target=myForward, args=(*inputs,))  #, daemon=True
                this_assigner.thread_pool[myGlobalTopo] = th
                th.start()
                this_assigner.no_threads_launched +=1
            else:
                _LOGGER.debug("Launching in main thread")
                myForward(*inputs)
            this_assigner.gpuwise_execution_status[module_gpu][myTopo]=1
            return result

        return modified_forward

    def recur_move_layers_to_gpus(self):    
        this_assigner = self
        #nodes are sorted topologically, to get the gpuwise topo_order of each module
        sorted_byTopo = dict(sorted(this_assigner.model.sub_module_nodes.items(), 
                                key=lambda x: x[1].topo_order)).keys()
        for module_id in sorted_byTopo:
            module_node = this_assigner.model.sub_module_nodes[module_id]
            module = module_node.module
            module_gpu = module_node.p
    
            ### Move layers to the allotted GPUs
            module.to(module_gpu)
            this_assigner.original_forwards[module_id] = module.forward
            this_assigner.output_record[module_id] = {}

            gpu_kernel_queue = this_assigner.gpuwise_execution_status[module_gpu]
            gpu_events_list = this_assigner.events[module_gpu]
            module_node.gpu_topo_order = len(gpu_kernel_queue) #topo order of the node on its gpu
            gpu_kernel_queue.append(0) # append a 0 - used later to indicate launch status
            this_assigner.gpuwise_queue[module_gpu].append(module_node)
            gpu_events_list.append(threading.Event()) # will be used to indicate completion of 
                                                      # this module's thread

            ####################---------Build necessary streams---------#####################
            for child_id in module_node.children:
                child_gpu = this_assigner.model.sub_module_nodes[child_id].p
                if (child_gpu != module_gpu) and (child_gpu not in module_node.output_stream):
                    ### tx and rx streams
                    module_node.output_stream[child_gpu] =\
                    {'tx':torch.cuda.Stream(device=module_gpu), 'rx':torch.cuda.Stream(device=child_gpu)}

            modified_forward = self.get_modified_forward(module_node, module_gpu, 
                                    module_node.gpu_topo_order, module_node.topo_order)
            module.forward =  modified_forward.__get__(module, module.__class__)
    
    def wait_for_threads(self, output):
        for th in self.thread_pool:
            if th is not None:
                th.join()
        self.thread_pool = [None for _ in range(len(self.model.sub_module_nodes))]
        for each_gpu in self.gpuwise_execution_status: # reset execution statuses and events
            L =  len(self.gpuwise_execution_status[each_gpu])
            self.gpuwise_execution_status[each_gpu] = [0 for _ in range(L) ] 
            self.events[each_gpu] = [threading.Event() for _ in range(L) ]
        self.wait_status = [True for _ in range(len(self.model.sub_module_nodes))]
        _LOGGER.info("Number of threads launched {} out of {} nodes".format(self.no_threads_launched, 
                            len(self.model.sub_module_nodes)) )
        return output[0] # strip out the mutable list used to pass outputs among threads

    def clear_records(self):  ## must be done after each forward
        self.own_outputs = {} 
        self.output_record = {}
        self.no_threads_launched = 0

#------------------------------------------------------------------------------------------#
############################################################################################

class Assign_global_topo(object):
    """
    Assigner that supports:
    - streams for cross-device communication
    - reordering of layer according to the global topo order (using threads)
    """
    def __init__(self, model_wrapper, args):
        self.model = model_wrapper
        self.original_forwards = {}
        self.own_outputs = {} #just to keep reference to recent outputs
        self.output_record = {} # module_ouput_id: gpu_id -> (pointer to copy of the 'output' at 'gpu')
        self.thread_pool = [None for _ in range(len(self.model.sub_module_nodes))]
        self.events = [threading.Event() for _ in range(len(self.model.sub_module_nodes))]
        self.execution_status = [0 for _ in range(len(self.model.sub_module_nodes))]
        self.assigned = self.recur_move_layers_to_gpus(model_wrapper.model)
        self.no_threads_launched = 0

        self.run_type   = args.run_type
        self.model_type = args.model_type
    
    def recur_move_layers_to_gpus(self, module):
        
        this_assigner = self
        sub_modules = module.__dict__['_modules']
        if len(sub_modules) > 0 and (module.__class__.__name__ not in basic_blocks):
            for name, sub_module in sub_modules.items():
                this_assigner.recur_move_layers_to_gpus(sub_module)
        elif (this_assigner.model_type  == 'gnmt' and module.__class__.__name__ =='Dropout'): #To exclude dropout layers (they need not be moved to gpus)
            pass
        else:
            module_id = id(module)
            module_node = this_assigner.model.sub_module_nodes[module_id]
            module_gpu = module_node.p
    
            ### Move layers to the allotted GPUs
            module.to(module_gpu)
            this_assigner.original_forwards[module_id] = module.forward
            this_assigner.output_record[module_id] = {}

            #####---------Build necessary streams---------#######
            for child_id in module_node.children:
                child_gpu = this_assigner.model.sub_module_nodes[child_id].p
                if (child_gpu != module_gpu) and (child_gpu not in module_node.output_stream):
                    ### tx and rx streams
                    module_node.output_stream[child_gpu] =\
                    {'tx':torch.cuda.Stream(device=module_gpu), 'rx':torch.cuda.Stream(device=child_gpu)}

            def modified_forward(self, *inputs):
                myTopo = module_node.topo_order
                result =[None]
                #########################################################
                def myForward(*inputs ):
                    #-------------------------------------------------------
                    if myTopo>0:
                        prev_event = this_assigner.events[myTopo-1]
                        prev_event.wait()
                    _LOGGER.debug("Launching:", module_node.module )
                    my_event = this_assigner.events[myTopo]
                    #-------------------------------------------------------
                    ### Wait for the rx streams from parent nodes to current gpu
                    for par_id in module_node.parent:
                        par_gpu    = this_assigner.model.sub_module_nodes[par_id].p
                        if par_gpu != module_gpu:
                            par_stream = this_assigner.model.sub_module_nodes[par_id].output_stream[module_gpu]['rx']
                            (torch.cuda.default_stream(device=module_gpu)).wait_stream(par_stream)
                    #-------------------------------------------------------
                    input_list = list(inputs)
                    for i, inp in enumerate(input_list):
                        if myTopo>0 and isinstance(inp, list): 
                            inp = inp[0] #to extract value from the list passed by previous thread
                        if torch.is_tensor(inp): #if input is tensor, get its copy on ther current GPU
                            inp_device = inp.get_device()
                            inp_id = str(inp_device) + '-' + str(id(inp))
                            if inp_device != module_gpu:
                                if inp_id in this_assigner.output_record:
                                    ## get the input's copy on current gpu
                                    input_list[i] = this_assigner.output_record[inp_id][module_gpu]
                                else: 
                                    # Lateral inputs (like masks in transformer), which are not output of some previous layer
                                    #print("A lateral input at", module_node.name)
                                    #assert (inp.get_device()<0) #Lateral inputs come from cpu
                                    input_list[i] = inp.to(module_gpu)
                            else:
                                input_list[i] = inp
                    inputs = tuple(input_list)
                    #-------------------------------------------------------
                    output = this_assigner.original_forwards[module_id](*inputs) 
                    result[0] = output
                    output_id = str(module_gpu) + '-' + str(id(output))
                    this_assigner.output_record[output_id] = {} ## for current output record pointers to its copies on child gpus
                            
                    if this_assigner.run_type == "inference":                      ## just to keep a reference to thhe output
                        this_assigner.own_outputs[module_id]=output ## leads to correctness issue without this
                                                                    ## where the memory (thus object ids) are reused
                    ## Make output tx Copy Streams wait for the Compute Stream
                    for child_id in module_node.children:
                        child_gpu = this_assigner.model.sub_module_nodes[child_id].p
                        if child_gpu != module_gpu:
                            this_assigner.output_record[output_id][child_gpu] = None
                            module_node.output_stream[child_gpu]['tx'].wait_stream(torch.cuda.default_stream(device=module_gpu))
                    
                    ## Start transferring the output to child gpus 
                    for child_id in module_node.children:
                        child_gpu = this_assigner.model.sub_module_nodes[child_id].p
                        if child_gpu != module_gpu:
                            if this_assigner.output_record[output_id][child_gpu] is None: 
                                #else output has already been transferred to child gpu
                                with torch.cuda.stream(module_node.output_stream[child_gpu]['tx']):
                                    with torch.cuda.stream(module_node.output_stream[child_gpu]['rx']):
                                        this_assigner.output_record[output_id][child_gpu] = output.to(child_gpu, non_blocking=True)
                    my_event.set()
                    return
                    ########################################################
                
                if myTopo>0:
                    if sum(this_assigner.execution_status[:myTopo])<myTopo: 
                        #need to spawn a thread, not all predecessors have been launched yet
                        th = threading.Thread(target=myForward, args=(*inputs,))  #, daemon=True
                        this_assigner.thread_pool[myTopo] = th
                        th.start()
                        this_assigner.no_threads_launched += 1
                    else:
                        myForward(*inputs)
                else:
                    myForward(*inputs)
                this_assigner.execution_status[myTopo]=1
                return result

            module.forward =  modified_forward.__get__(module, module.__class__)
    
    def wait_for_threads(self, output):
        for th in self.thread_pool:
            if th is not None:
                th.join()
        self.execution_status = [0 for _ in range(len(self.model.sub_module_nodes))]
        self.thread_pool = [None for _ in range(len(self.model.sub_module_nodes))]
        self.events = [threading.Event() for _ in range(len(self.model.sub_module_nodes))] 
        _LOGGER.info("Number of threads launched {} out of {} nodes".format(self.no_threads_launched, 
                            len(self.model.sub_module_nodes)) )
        return output[0]

    def clear_records(self):  ## must be done after each forward
        self.own_outputs = {} 
        self.output_record = {}
        self.no_threads_launched = 0


#------------------------------------------------------------------------------------------#
############################################################################################
class Assign_plain(object):
    """
    Assigner
    - uses plain '.to(device)'
    - no streams for communication
    - no reordering of layers

    """
    def __init__(self, model_wrapper, args):
        self.model = model_wrapper
        self.original_forwards = {}
        self.assigned = self.recur_move_layers_to_gpus(model_wrapper.model)

        self.run_type   = args.run_type
        self.model_type = args.model_type
    
    def recur_move_layers_to_gpus(self, module):
        
        this_assigner = self
        sub_modules = module.__dict__['_modules']
        if len(sub_modules) > 0 and (module.__class__.__name__ not in basic_blocks):
            for name, sub_module in sub_modules.items():
                this_assigner.recur_move_layers_to_gpus(sub_module)
        elif (this_assigner.model_type  == 'gnmt' and module.__class__.__name__ =='Dropout'): #To exclude dropout layers (they need not be moved to gpus)
            pass
        else:
            module_id = id(module)
            module_gpu = this_assigner.model.sub_module_nodes[module_id].p
    
            ### Move layers to the allotted GPUs
            module.to(module_gpu)
            this_assigner.original_forwards[module_id] = module.forward

            def modified_forward(self, *inputs):
                #########################################################
                input_list = list(inputs)
                for i, inp in enumerate(input_list):
                    input_list[i] = inp.to(module_gpu)
                inputs = tuple(input_list)
                ########################################################
                output = this_assigner.original_forwards[module_id](*inputs) 
                return output

            module.forward =  modified_forward.__get__(module, module.__class__) 

    def wait_for_threads(self,output): #dummy
        return output

    def clear_records(self):  #dummy
        return

#------------------------------------------------------------------------------------------#
############################################################################################
class Assign_stream(object):
    """
    Assinger:
    - Uses streams for cross-device comm
    - But does no reordering of layer
    """
    def __init__(self, model_wrapper, args):
        self.model = model_wrapper
        self.original_forwards = {}
        self.own_outputs = {} #just to keep reference to recent outputs
        self.output_record = {} # module_ouput_id: gpu_id -> (pointer to copy of the 'output' at 'gpu')
        self.assigned = self.recur_move_layers_to_gpus(model_wrapper.model) # this must be
        # the last call on __init__. Else recur_... can 't use the what's defined after this line

        self.run_type   = args.run_type
        self.model_type = args.model_type
        
    def recur_move_layers_to_gpus(self, module):
        
        this_assigner = self
        sub_modules = module.__dict__['_modules']
        if len(sub_modules) > 0 and (module.__class__.__name__ not in basic_blocks):
            for name, sub_module in sub_modules.items():
                this_assigner.recur_move_layers_to_gpus(sub_module)
        elif (this_assigner.model_type  == 'gnmt' and module.__class__.__name__ =='Dropout'): #To exclude dropout layers (they need not be moved to gpus)
            pass
        else:
            module_id   = id(module)
            module_node = this_assigner.model.sub_module_nodes[module_id]
            module_gpu  = module_node.p

            ### Move layers to the allotted GPUs
            module.to(module_gpu)
            this_assigner.original_forwards[module_id] = module.forward
            this_assigner.output_record[module_id] = {}
            
            #####---------Build necessary streams---------#######
            for child_id in module_node.children:
                child_gpu = this_assigner.model.sub_module_nodes[child_id].p
                if (child_gpu != module_gpu) and (child_gpu not in module_node.output_stream):
                    ### tx and rx streams
                    module_node.output_stream[child_gpu] =\
                    {'tx':torch.cuda.Stream(device=module_gpu), 'rx':torch.cuda.Stream(device=child_gpu)}


            def modified_forward(self, *inputs):
                
                ### Wait for the rx streams from parent nodes to current gpu
                for par_id in module_node.parent:
                    par_gpu    = this_assigner.model.sub_module_nodes[par_id].p
                    if par_gpu != module_gpu:
                        par_stream = this_assigner.model.sub_module_nodes[par_id].output_stream[module_gpu]['rx']
                        (torch.cuda.default_stream(device=module_gpu)).wait_stream(par_stream)
                
                #########################################################
                input_list = list(inputs)
                for i, inp in enumerate(input_list):
                    inp_device = inp.get_device()
                    inp_id = str(inp_device) + '-' + str(id(inp))
                    if inp_device != module_gpu:
                        ## get the input's copy on current gpu
                        if inp_id in this_assigner.output_record:
                            input_list[i] = this_assigner.output_record[inp_id][module_gpu]
                        else:# Lateral inputs (like masks in transformer), which are not output of some previous layer
                            #print("A lateral input at", module_node.name)
                            #assert (inp.get_device()<0) #Lateral inputs come from cpu
                            input_list[i] = inp.to(module_gpu)

                inputs = tuple(input_list)
                output = this_assigner.original_forwards[module_id](*inputs) 
                output_id = output_id = str(module_gpu) + '-' + str(id(output))
                this_assigner.output_record[output_id] = {} ## for current output record pointers to its copies on child gpus
                        
                if this_assigner.run_type == "inference":                      ## just to keep a reference to thhe output
                    this_assigner.own_outputs[module_id]=output ## leads to correctness issue without this
                                                                ## where the memory (thus object ids) are reused
                
                ## Make output tx Copy Streams wait for the Compute Stream
                for child_id in module_node.children:
                    child_gpu = this_assigner.model.sub_module_nodes[child_id].p
                    if child_gpu != module_gpu:
                        this_assigner.output_record[output_id][child_gpu] = None
                        module_node.output_stream[child_gpu]['tx'].wait_stream(torch.cuda.default_stream(device=module_gpu))
                
                ## Start transferring the output to child gpus 
                for child_id in module_node.children:
                    child_gpu = this_assigner.model.sub_module_nodes[child_id].p
                    if child_gpu != module_gpu:
                        if this_assigner.output_record[output_id][child_gpu] is None: 
                            #else output has already been transferred to child gpu
                            with torch.cuda.stream(module_node.output_stream[child_gpu]['tx']):
                                with torch.cuda.stream(module_node.output_stream[child_gpu]['rx']):
                                    this_assigner.output_record[output_id][child_gpu] = output.to(child_gpu, non_blocking=True)

                return output
                ## If each node has all children on just one gpu, then we can output the
                ## copy on the child gpu instead of 'output'(which is in the current gpu) 
            
            module.forward =  modified_forward.__get__(module, module.__class__)  
            
    def clear_records(self):  ## must be done after each forward
        self.own_outputs = {} 
        self.output_record = {}

    def wait_for_threads(self, output): #dummy
        return output
#------------------------------------------------------------------------------------------#
############################################################################################

class Assign_thread(object):
    """
    Does not use streams for comm. Uses blocking '.to()'
    Also uses cumulative of execution_order-topo_order differences to launch (not optimal)
    """
    def __init__(self, model_wrapper, args):
        self.model = model_wrapper
        self.original_forwards = {}
        self.thread_pool = [None for _ in range(len(self.model.sub_module_nodes))]
        self.events = [threading.Event() for _ in range(len(self.model.sub_module_nodes))]
        self.cumulative_diff = 0 # cumulative of execution_order-topo_order
        self.assigned = self.recur_move_layers_to_gpus(model_wrapper.model)

        self.run_type   = args.run_type
        self.model_type = args.model_type
    
    def recur_move_layers_to_gpus(self, module):
        
        this_assigner = self
        sub_modules = module.__dict__['_modules']
        if len(sub_modules) > 0 and (module.__class__.__name__ not in basic_blocks):
            for name, sub_module in sub_modules.items():
                this_assigner.recur_move_layers_to_gpus(sub_module)
        elif (this_assigner.model_type  == 'gnmt' and module.__class__.__name__ =='Dropout'): #To exclude dropout layers (they need not be moved to gpus)
            pass
        else:
            module_id = id(module)
            module_node = this_assigner.model.sub_module_nodes[module_id]
            module_gpu = module_node.p
    
            ### Move layers to the allotted GPUs
            module.to(module_gpu)
            this_assigner.original_forwards[module_id] = module.forward

            def modified_forward(self, *inputs):
                myTopo = module_node.topo_order
                #myTopo = module_node.execution_order   #not topo order, actual execution order
                this_assigner.cumulative_diff = this_assigner.cumulative_diff + (module_node.execution_order - module_node.topo_order)
                result =[None]
                #########################################################
                def myForward(*inputs ):
                    if myTopo>0:
                        prev_event = this_assigner.events[myTopo-1]
                        prev_event.wait()
                    #print("Launching:", myTopo)
                    my_event = this_assigner.events[myTopo]
                    input_list = list(inputs)
                    for i, inp in enumerate(input_list):
                        if myTopo>0 and isinstance(inp, list):
                            #print(module_node.module)
                            #print(len(inp))
                            #print(inp[0].size())
                            #print("*"*20)
                            input_list[i] = inp[0].to(module_gpu)
                        else:
                            input_list[i] = inp.to(module_gpu)
                    inputs = tuple(input_list)
                    result[0] = this_assigner.original_forwards[module_id](*inputs)
                    my_event.set()
                    return
                    ########################################################
                if this_assigner.cumulative_diff < 0:
                    th = threading.Thread(target=myForward, args=(*inputs,))  #, daemon=True
                    this_assigner.thread_pool[myTopo] = th
                    th.start()
                else:
                    myForward(*inputs)
                return result

            module.forward =  modified_forward.__get__(module, module.__class__)
    
    def wait_for_threads(self, output):
        for th in self.thread_pool:
            if th is not None:
                th.join()
        self.cumulative_diff = 0
        self.thread_pool = [None for _ in range(len(self.model.sub_module_nodes))]
        self.events = [threading.Event() for _ in range(len(self.model.sub_module_nodes))]
        return output[0]

    def clear_records(self):  #dummy
        return
#------------------------------------------------------------------------------------------#
############################################################################################
class Assign_gnmt_stream(object):
    """
    Assinger:
    - Uses streams for cross-device comm
    - But does no reordering of layer
    """
    def __init__(self, model_wrapper, args):
        self.model = model_wrapper
        self.original_forwards = {}
        self.own_outputs = {} #just to keep reference to recent outputs
        self.output_record = {} # module_ouput_id: gpu_id -> (pointer to copy of the 'output' at 'gpu')
        self.assigned = self.recur_move_layers_to_gpus(model_wrapper.model) # this must be
        # the last call on __init__. Else recur_... can 't use the what's defined after this line
        self.run_type   = args.run_type
        self.model_type = args.model_type
        
    def recur_move_layers_to_gpus(self, module):
        
        this_assigner = self
        sub_modules = module.__dict__['_modules']
        if len(sub_modules) > 0 and (module.__class__.__name__ not in basic_blocks):
            for name, sub_module in sub_modules.items():
                this_assigner.recur_move_layers_to_gpus(sub_module)
        elif (this_assigner.model_type  == 'gnmt' and module.__class__.__name__ =='Dropout'): #To exclude dropout layers (they need not be moved to gpus)
            pass
        else:
            module_id   = id(module)
            module_node = this_assigner.model.sub_module_nodes[module_id]
            module_gpu  = module_node.p

            ### Move layers to the allotted GPUs
            module.to(module_gpu)

            if module_id not in this_assigner.original_forwards:
                
                this_assigner.original_forwards[module_id] = module.forward
                this_assigner.output_record[module_id] = {}
                
                #####---------Build necessary streams---------#######
                for child_id in module_node.children:
                    child_gpu = this_assigner.model.sub_module_nodes[child_id].p
                    if (child_gpu != module_gpu) and (child_gpu not in module_node.output_stream):
                        ### tx and rx streams
                        module_node.output_stream[child_gpu] =\
                        {'tx':torch.cuda.Stream(device=module_gpu), 'rx':torch.cuda.Stream(device=child_gpu)}


                def modified_forward(self, *inputs):
                    
                    ### Wait for the rx streams from parent nodes to current gpu
                    for par_id in module_node.parent:
                        par_gpu    = this_assigner.model.sub_module_nodes[par_id].p
                        if par_gpu != module_gpu:
                            par_stream = this_assigner.model.sub_module_nodes[par_id].output_stream[module_gpu]['rx']
                            (torch.cuda.default_stream(device=module_gpu)).wait_stream(par_stream)
                    
                    #########################################################
                    input_list = list(inputs)
                    #print("----"*50)
                    print(module)
                    print(module_gpu)
                    for i, inp in enumerate(input_list):                        
                        diff_device = False #indicates if tensor is on a different device
                        if (isinstance(inp, torch.Tensor)):
                            inp_device = inp.get_device()
                            diff_device = ( inp_device != module_gpu)
                            if diff_device:
                                inp_id = str(inp_device) + "-" +str(id(inp)) +'-'+ str(get_first_element(inp))
                        elif (type(inp) is torch.nn.utils.rnn.PackedSequence):
                            inp_device = inp.data.get_device()
                            diff_device = ( inp_device != module_gpu)
                            if diff_device:
                                inp_id = str(inp_device) + "-" +str(id(inp))+'-'+ str(get_first_element(inp.data))
                        elif isinstance(inp, tuple):
                             raise Exception("tuples not handled")

                        if diff_device:
                            
                            print("Input_ID:", inp_id)
                            ## get the input's copy on current gpu
                            if inp_id in this_assigner.output_record:
                                if module_gpu in this_assigner.output_record[inp_id]:
                                    print("Found already!")
                                    input_list[i] = this_assigner.output_record[inp_id][module_gpu]
                                else:
                                    
                                    print("Input in record but lateral:module_id: ",(module_id), module_gpu, inp_id)
                                    #try: # TODO: Has correctness issue:
                                    #    print(inp[0][0][0]) # this value does not match an earlier output with same id
                                    #except:
                                    #    pass
                                    #print(this_assigner.output_record[inp_id])
                                    input_list[i] = inp.to(module_gpu) # handle lateral parameter like 'self.linear_att'
                            else:# Lateral inputs (like masks in transformer), which are not output of some previous layer
                                #print("A lateral input at", module_node.name)
                                #assert (inp.get_device()<0) #Lateral inputs come from cpu
                                print(module)
                                print("Input is lateral: module_id: ",module_id, module_gpu, inp_id)
                                inp_transferred = inp.to(module_gpu)
                                input_list[i] = inp_transferred
                                this_assigner.output_record[inp_id] ={}
                                this_assigner.output_record[inp_id][module_gpu] = inp_transferred

                    inputs = tuple(input_list)
                    output_raw = this_assigner.original_forwards[module_id](*inputs)
                    if isinstance(output_raw, tuple):
                        output_dict = recursively_get_output_ids(output_raw, module_gpu)
                    else:
                        #print("TYPE of output raw::", type(output_raw))
                        output_id = str(module_gpu) + "-" + str(id(output_raw)) +'-'+ str(get_first_element(output_raw))  # two tensors on different devices may have same id. So append gpu_id to id
                        output_dict={}
                        output_dict[output_id] = output_raw

                       
                    
                    for output_id in  output_dict: 
                        print("-"*10) 
                        print("output_ID:",output_id)         
                        this_assigner.output_record[output_id] = {} ## for current output record pointers to its copies on child gpus
                        #print("Fill:", module_id, output_id)
                        #try:
                        #    print(output_dict[output_id][0][0][0])
                        #except:
                        #    pass
                    print("-"*70)
                    print()

                    if this_assigner.run_type == "inference":        
                        this_assigner.own_outputs[module_id] = {} 
                        for output_id in  output_dict:                    ## just to keep a reference to thhe output
                            this_assigner.own_outputs[module_id][output_id]=output_dict[output_id] ## leads to correctness issue without this
                                                                    ## where the memory (thus object ids) are reused
                    
                    ## Make output tx Copy Streams wait for the Compute Stream
                    for child_id in module_node.children:
                        child_gpu = this_assigner.model.sub_module_nodes[child_id].p
                        child_mod = this_assigner.model.sub_module_nodes[child_id].module
                        #print(child_mod, id(child_mod))
                        if child_gpu != module_gpu:
                            for output_id in  output_dict:
                                #print("Set:", module_id, output_id, child_id)
                                this_assigner.output_record[output_id][child_gpu] = None
                            module_node.output_stream[child_gpu]['tx'].wait_stream(torch.cuda.default_stream(device=module_gpu))
                    #print("*"*70)
                    #print()
                    ## Start transferring the output to child gpus 
                    for child_id in module_node.children:
                        child_gpu = this_assigner.model.sub_module_nodes[child_id].p
                        if child_gpu != module_gpu:
                            for output_id in  output_dict:
                                if this_assigner.output_record[output_id][child_gpu] is None: 
                                    #else output has already been transferred to child gpu
                                    with torch.cuda.stream(module_node.output_stream[child_gpu]['tx']):
                                        with torch.cuda.stream(module_node.output_stream[child_gpu]['rx']):
                                            this_assigner.output_record[output_id][child_gpu] = output_dict[output_id].to(child_gpu, non_blocking=True)
                    return output_raw
                    ## If each node has all children on just one gpu, then we can output the
                    ## copy on the child gpu instead of 'output'(which is in the current gpu) 
                
                module.forward =  modified_forward.__get__(module, module.__class__)  
            
    def clear_records(self):  ## must be done after each forward
        self.own_outputs = {} 
        self.output_record = {}

    def wait_for_threads(self, output): #dummy
        return output
#------------------------------------------------------------------------------------------#
############################################################################################
class Assign_gnmt_plain(object):
    """
    This class actually put each submodule to the gpu it is assigned to
    """
    def __init__(self, model_wrapper, args):
        self.model = model_wrapper
        self.original_forwards = {}
        self.assigned = self.recur_move_layers_to_gpus(model_wrapper.model)
        self.run_type   = args.run_type
        self.model_type = args.model_type
    
    def recur_move_layers_to_gpus(self, module):
        
        this_assigner = self
        sub_modules = module.__dict__['_modules']
        mod_name = str(module)[0:4]
        if len(sub_modules) > 0:
            for name, sub_module in sub_modules.items():
                this_assigner.recur_move_layers_to_gpus(sub_module)
        elif (mod_name =='Drop'):# or (mod_name =='_add'):
            pass
        else:    
            module_id = id(module)
            gpu_id = this_assigner.model.sub_module_nodes[module_id].p

            ### Move layers to the allotted GPUs
            # module.to(gpu_id)
            ########### FOR TESTING ##################################################
            module.to(gpu_id)
            #########################################################################
            
            if module_id not in this_assigner.original_forwards:
                this_assigner.original_forwards[module_id] = module.forward

                def modified_forward(self, *inputs):
                    print(module,"********")
                    print(gpu_id)
                    input_list = list(inputs)
                    for i, inp in enumerate(input_list):
                        if (isinstance(inp, torch.Tensor))  or (type(inp) is torch.nn.utils.rnn.PackedSequence):
                            input_list[i] = inp.to(gpu_id)
                        else:
                            print(inp)
                            pass
                    inputs = tuple(input_list)
                    output = this_assigner.original_forwards[module_id](*inputs) 
                    return output

                module.forward =  modified_forward.__get__(module, module.__class__) 

    def wait_for_threads(self, output):#dummy
        return output

    def clear_records(self):  #dummy
        return
#####################################################################################################

def assigner(model_wrapper, assigner_type, args):
    
    if assigner_type == 'gpu_topo_based':
        _LOGGER.info("Assigner choosen: GPU topo based reordering + streams based communication")
        assigner_handle = Assign_gpu_topo(model_wrapper, args)

    elif assigner_type == 'global_topo_based':
        _LOGGER.info("Assigner choosen: Global topo based reordering + streams based communication" )
        assigner_handle = Assign_global_topo(model_wrapper, args)

    elif assigner_type == 'only_stream_no_reordering':
        _LOGGER.info("Assigner choosen: No reordering of layers according to topo order." +
                        " Only streams for communication" )
        assigner_handle = Assign_stream(model_wrapper, args)

    elif assigner_type == 'no_stream_no_reordering':
        _LOGGER.info("Assigner choosen: Plain - no streams, no reordering. " + 
                        "Blocking '.to()' is used" )
        assigner_handle = Assign_plain(model_wrapper, args)

    elif assigner_type == 'no_stream_only_reordering':
        _LOGGER.info("Assigner choosen: Only reordering of layers. No streams (mostly useless :p)" )
        assigner_handle = Assign_thread(model_wrapper, args)

    elif assigner_type == 'gnmt_assigner_plain':
        _LOGGER.info("Assigner choosen: for GNMT" )
        assigner_handle = Assign_gnmt_plain(model_wrapper, args)

    elif assigner_type == 'gnmt_assigner_stream':
        _LOGGER.info("Assigner choosen: for GNMT" )
        assigner_handle = Assign_gnmt_stream(model_wrapper, args)
    else:
        raise Exception("Invalid assigner type:", assigner_type)

    return assigner_handle

