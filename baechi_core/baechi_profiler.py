import torch
import time
from torch import optim, nn
#import numpy as np

from utils import logger
import util_functions as utilsf
import config

_LOGGER = logger.get_logger(__file__, level=logger.INFO)

class SubModuleNode:
    """
    This class represents a submodel (ex. conv2d layer) in the given model (ex. inception_v3). 
    It is represented as a node in the return graph
    """
    def __init__(self):
        # store the entire submodel
        self.module = None
        # submodel name
        self.name = None

        # nodes that must finish processing before this node (direct dependencies)
        self.parent = set()
        # nodes that depends on this node
        self.children = set()

        # forward function's estimated runtime
        self.weight_forward = 0
        # backward function's estimated runtime
        self.weight_backward = 0
        # id represented by the model's location (python's id function)
        self.id_hash = None
        # sudo id used, for one model, this sudo id starts from 0 and add 1 for each new node
        self.id = None
        #------------------- Memory parameters ----------------------
        # storage used by submodel's parameters and their gradients
        self.parameter_mem = 0
        self.parameter_grad_mem = 0
        # submodel's input's size
        self.input_mem= 0
        # submodel's output's size
        self.output_mem = 0
        # submodel's net mem used in forward
        #  = output_mem for single layer, 
        #  = output_mem + intermediate outputs for multiple layer blocks
        self.net_used_mem = 0
        # submodel's temporary usage during computation
        self.temp_compute_mem = 0

        # permanent storage: see "update_permanent_and_temporary_mem"
        self.permanent_mem = 0
        # temporary memory
        self.temporary_mem = 0
        # peak = permanent + temporary
        self.peak_mem = 0

        self.inplace = False # A flag to indicate inplace op. i.e cause no memory change
         #------------------- Assigner parameters ----------------------
        # topo_order in final execution sequence
        self.topo_order = None
        self.gpu_topo_order = None #topo order within the asigned gpu
        self.execution_order = None
        
        # gpu assigned to the submodule
        self.p = None

        # Stream used to transfer the output out of the node to child nodes
        # on other gpus
        self.output_stream = {}


    def update_permanent_and_temporary_mem(self, itype, model_type):
        ##########################################################################################
        #--------- Inputs ----------
        # node is object of type SubModuleNode
        # itype is either inference or training
        # model_type is either 'gnmt, 'inception', or 'transformer'

        #--------- Permanent and Temporary memory --------
        # if Training:
        # permanent_mem: node's parameters, their gradients and intermediate outputs
        # temporary_mem: during backward the upstream grad + downstream grad + the parameter 
        #                grads(for a moment both the new and previous parameter grad coexist)
        #               + temp mem used during the computation
        # Note 1: upstream grad size = output size, downstream grad size = input size
        # Note 2: But if the modules is made of multiple modules (like those listed in config.BASIC_BLOCKS),
        #         ,then we must use the largest intermediate downstream grad size within the block - we
        #         estimate this as max(self.net_used_mem-self.output_mem, self.input_mem), where 
        #         (self.net_used_mem-self.output_mem) is the sum of sizes of all intermediate outputs.
        #         If the module has many intermediate modules, then this may lead to large overestimation
        #         of memory. TODO: compute more accurarte memory needs of multi-module modules 
        #
        # if Inference:
        # permanent_mem: node's parameters
        # temporary_mem: storage for input, output and temp mem used in compute
        #
        # peak_mem =  permanent_mem + temporary_mem
        #
        #--------- Placement feasibility check ----------
        # (In placer.device.is_feasible())
        # To place a node on a device D, check:
        # 1. D_used_mem + node.peak_mem < D_max_mem       # scenario when node is executing
        # 2. D_peak_mem + node.parameter_mem < D_max_mem  # scenario when some other node on D is executing
        # (where, D_peak_mem is peak memory usage on D until placing this node)
        #
        # If the check passes, place the node on D and update D_used_mem and D_peak_mem:
        # 1. D_max_temp_mem = (D_peak_mem - D_used_mem)
        #    If  D_max_temp_mem < node.temporary_mem:
        #        D_max_temp_mem = node.temporary_mem     
        # 2. D_used_mem = D_used_mem + node.permanent_mem
        # 3. D_peak_mem = D_used_mem + D_max_temp_mem
        #####################################################################################################

        if itype== "training":
            # Exceptions for [_concatenate, inplace Relu]
            if self.name[:4]=="_con": #Concatenate
                self.permanent_mem =  self.output_mem
                self.temporary_mem =  0 # concatenate needs no other memory during output
            elif (self.name[:4]=="ReLU" or self.name[:4]=="_fla" or self.name[:4]=="_squ" ) and self.inplace: 
                #relu or _flatten() or _squeez() 
                self.permanent_mem = 0; self.temporary_mem = 0
            elif (model_type  == 'gnmt') and (self.name[:4]=="_add" or self.name[:4]=="_tan" or self.name[:4]=="_mas"): 
                # for _addLayer, _tanhLayer or _maskedFill() in gnmt
                #TODO: currently restricted to gnmt. Should be fine for _addLayer in other models too
                self.permanent_mem = self.output_mem
                self.temporary_mem = self.output_mem
            elif (self.name[:4]=="MaxP"): #maxpool
                # (Unexaplined: Maxpool forward peak takes a net_used of 3*output_mem
                # but only takes 2*output_mem by the end of entire models forward)
                self.permanent_mem = self.net_used_mem 
                assert self.permanent_mem>=0
                self.temporary_mem =  0
            else:
                # Default
                self.permanent_mem = self.parameter_mem + self.parameter_grad_mem + self.net_used_mem # (self.net_used_mem includes self.output_mem)
                self.temporary_mem =  self.output_mem + self.parameter_grad_mem + max(self.net_used_mem-self.output_mem, self.input_mem) + self.temp_compute_mem

        
        elif itype== "inference":
            if self.name[:4]=="_con": #Concatenate
                self.permanent_mem = 0 ; self.temporary_mem =  self.output_mem
            elif (self.name[:4]=="ReLU" or self.name[:4]=="_fla" or self.name[:4]=="_squ") and self.inplace:
                self.permanent_mem = 0 ; self.temporary_mem =  0
            else:
                self.permanent_mem = self.parameter_mem 
                self.temporary_mem = self.input_mem + self.temp_compute_mem + self.net_used_mem
        else:
            raise ValueError("Run type should be either inference or training")

        self.peak_mem = self.permanent_mem + self.temporary_mem
        
########################################################################

class Profiling:
    """
    This class produce the profile, this class referenced "https://github.com/msr-fiddle/pipedream"
    """
    def __init__(self, run_type, model, model_type, gpu=0, rounds=20, input_size=(3, 299, 299), output_size = 1000, batch_size = 32, model_info={}):
        """
        model: ex. inception_v3 model, transformer etc
        gpu: choose in between {0,1,2,3}
        rounds: number of rounds to run the profiling
        """
        self.run_type = run_type
        self.gpu = gpu
        self.model = model.to(self.gpu)
        self.model_type = model_type
        self.input_size = input_size
        self.output_size = output_size
        self.batch_size = batch_size
        self.model_info = model_info

        self.rounds = rounds
        # first few rounds are inaccurate, so I choose to discard the results from the first 1/4 rounds
        self.ignore_rounds = int(self.rounds/4)
        # counting variable, runs from 0 - self.rounds
        self.cur_round = 0

        # used to calculate backward runtime for each submodule
        self.back_record = []
        # all submodules record of the form {id of the layer(submodule) : SubModuleNode created out of tha layer}
        self.sub_module_nodes = {}
        # use id_hash to record the order of submodules's execution
        self.submodule_order = []

        # internal use only, record the original forward functions for submodules
        self.forward_original_methods = {}
        # list of all forward times for each submodule
        self.forward_runtimes = {}
        # internal use only, switch back to the original forward functions after profiling
        self.detach_record = set()
        # Collect handles to all hooks added, so as to remove them in detach()
        self.hook_handles = []

        self.layer_no = None #used to fill the execution_order go each submodule

    def _calculate_time(self,function, *input):
        """
        - Helper function in forward wrapper
        - Calculates forward runtime
        """
        torch.cuda.synchronize(self.gpu)
        start_time = time.time()
        result = function(*input)
        torch.cuda.synchronize(self.gpu)
        stop_time = time.time()
        return (stop_time - start_time) * 1000 , result

    def _calculate_memory(self,function, *input):
        """
        - Helper function in forward wrapper
        - Calculates peak memory and static memory usage
        """
        with utilsf.TorchTracemalloc(self.gpu) as tt:
            torch.cuda.synchronize(self.gpu)
            start_time = time.time()
            result = function(*input)
            torch.cuda.synchronize(self.gpu)
            stop_time = time.time()

        inplace_op = False  # indicates if the operation was inplace
        if (tt.begin == tt.end) and (tt.peak == tt.end):
            inplace_op = True
        return (stop_time - start_time) * 1000, tt.used, tt.peaked , inplace_op, result

    def recur_function(self, module):
        """
        modify self.model: adding forward timing, backward timing, input output sizes, etc
        :param module: the model to recursively add forward/backward wrappers to
        """
        this_profiler = self
        sub_modules = module.__dict__['_modules']
        for name, sub_module in sub_modules.items():
            # sub modules of sub_module, if there are more than 1, we need further recursion
            sub_sub_modules = sub_module.__dict__['_modules']
            if len(sub_sub_modules) > 0 and (sub_module.__class__.__name__ not in config.BASIC_BLOCKS):
                self.recur_function(sub_module)
                continue
            if (self.model_type  == 'gnmt' and sub_module.__class__.__name__ =='Dropout'):  #To exclude dropout layers (they need not be moved to gpus)
                continue

            def forward_wrapper(cur_module, *input):
                """
                use this wrapper to replace the original forward function in submodules
                :param cur_module: the input submodule
                """
                # original forward function
                function = this_profiler.forward_original_methods[cur_module]

                if this_profiler.cur_round < this_profiler.ignore_rounds:
                    if this_profiler.cur_round == 0: 
                        # Record memory usage in the very first round
                        # record submodule execution order only in the first round
                        _LOGGER.info("\nProfiling module: \n {} \n ".format(cur_module) )
                        this_profiler.submodule_order.append(id(cur_module))
                        forward_time, used_mem, peaked_mem, self.inplace, result = this_profiler._calculate_memory(function, *input)

                        ## Input size in bytes
                        input_mem = 0
                        for inp in input:
                            input_mem = input_mem + utilsf.estimate_tensor_size(inp, 'B')
                        ## Module size in bytes
                        module_mem  = utilsf.estimate_model_size(cur_module,'B', False)
                        output_mem = utilsf.estimate_tensor_size(result, 'B')

                        if self.inplace:
                            _LOGGER.info("This module is inplace")
                        else:
                            if used_mem<=0: # eg: _maskedFill in gnmt
                                used_mem = 0
                                _LOGGER.info("Module has 0 or negative used_mem. It uses some temp mem, but no used mem")
                            else:
                                assert used_mem >= output_mem, "Memory estimate wrong"
                                if used_mem != output_mem:
                                    _LOGGER.info("Used mem is more than the output mem. node is likely made of multiple layers or is a MaxPool")

                        # record a SubModuleNode for each model layer
                        if id(cur_module) not in this_profiler.sub_module_nodes:
                            cur_node = SubModuleNode()
                            cur_node.id_hash = id(cur_module)
                            cur_node.id = len(this_profiler.submodule_order)-1 #topo_order in the code
                            cur_node.module = cur_module
                            cur_node.name = cur_module.__class__.__name__
                            cur_node.execution_order = this_profiler.layer_no
                            this_profiler.layer_no = this_profiler.layer_no+1                            
                            ########### Memory parameters ###############
                            cur_node.parameter_mem = module_mem
                            cur_node.parameter_grad_mem = cur_node.parameter_mem # TODO: Any Exceptions?
                            cur_node.input_mem  = input_mem
                            cur_node.output_mem = output_mem
                            cur_node.net_used_mem = used_mem 
                            cur_node.temp_compute_mem = peaked_mem - used_mem
                            cur_node.update_permanent_and_temporary_mem(self.run_type, self.model_type)
                            #############################################
                        else:
                            if (cur_module.__class__.__name__ =='Embedding') : # embedding are reused in encorder, decoder usually
                                cur_node = this_profiler.sub_module_nodes[id(cur_module)]
                            else:
                                raise Exception("Node already exists and not an embedding. Something wrong!")
                        
                        _LOGGER.info(  ( "\n\t module id: {} \n\t input mem: {} \n\t " + 
                        "parameter mem: {} \n\t output mem: {} \n\t temp compute mem: {} \n\t " +
                        "net used mem: {} \n\t peak mem: {} \n\t ---------- \n\t " +
                        "Permanent mem: {} \n\t Temporary mem: {} \n" + "-*"*40  ).format(
                            id(cur_module), utilsf.humanize_num_bytes(cur_node.input_mem),
                            utilsf.humanize_num_bytes(cur_node.parameter_mem),
                            utilsf.humanize_num_bytes(cur_node.output_mem),
                            utilsf.humanize_num_bytes(cur_node.temp_compute_mem),
                            utilsf.humanize_num_bytes(used_mem),
                            utilsf.humanize_num_bytes(peaked_mem),
                            utilsf.humanize_num_bytes(cur_node.permanent_mem),
                            utilsf.humanize_num_bytes(cur_node.temporary_mem)
                        ) )

                        this_profiler.sub_module_nodes[id(cur_module)] = cur_node
                        #this_profiler.forward_runtimes[id(cur_module)] = []
                    # do not record first few rounds
                    else:
                        result = function(*input)
                    return result
                
                ## collect relevant information of cur module
                forward_time, result = this_profiler._calculate_time(function, *input)
                
                cur_node = this_profiler.sub_module_nodes[id(cur_module)]
                # we want weight_forward as the average forward runtime of the relevent rounds
                cur_node.weight_forward += forward_time / (this_profiler.rounds - this_profiler.ignore_rounds)
                #this_profiler.forward_runtimes[id(cur_module)].append(forward_time) 
                return result

            def hook(cur_module, inputs, output):
                # this is for retriving the module inside make dot function
                if type(output) is tuple: ## required for LSTM unit outputs in gnmt
                    for out_elem in output:
                        # LSTM outputs are (output, (h,c))
                        if type(out_elem) is torch.nn.utils.rnn.PackedSequence: #'output' is processed here
                            out_elem.data.grad_fn.metadata['module'] = cur_module
                        else: #(h,c) is processed here
                            hook(cur_module, inputs, out_elem)
                else:
                    output.grad_fn.metadata['module'] = cur_module
                

            def backward_post_hook(cur_module, input, output):
                """
                add backward hook to record backward runtime
                :param cur_module: the input submodule
                """
                if this_profiler.cur_round < this_profiler.ignore_rounds:
                    # do not record first few rounds
                    return
                torch.cuda.synchronize(self.gpu)
                cur_time = time.time() * 1000
                this_profiler.back_record.append((id(cur_module), cur_time))

            if sub_module in self.forward_original_methods:
                # only record the original forward functions once
                continue
            self.forward_original_methods[sub_module] = sub_module.forward
            sub_module.forward = forward_wrapper.__get__(sub_module, sub_module.__class__)
            fhook_handle = sub_module.register_forward_hook(hook)
            bhook_handle =  sub_module.register_backward_hook(backward_post_hook)
            this_profiler.hook_handles.append(fhook_handle)
            this_profiler.hook_handles.append(bhook_handle)
            
            
    def detach(self, module):
        """
        use this helper function to detach all forward wrappers
        """
        this_profiler = self
        sub_modules = module.__dict__['_modules']
        for name, sub_module in sub_modules.items():
            sub_sub_modules = sub_module.__dict__['_modules']
            if len(sub_sub_modules) > 0 and (sub_module.__class__.__name__ not in config.BASIC_BLOCKS):
                self.detach(sub_module)
                continue
            if sub_module in self.detach_record:
                continue
            if (self.model_type  == 'gnmt' and sub_module.__class__.__name__ =='Dropout'): #To exclude dropout layers (they need not be moved to gpus)
                continue

            self.detach_record.add(sub_module)
            sub_module.forward = self.forward_original_methods[sub_module]

            #runtimes = self.forward_runtimes[id(sub_module)]
            #mean_time = np.mean(runtimes)
            #std_time = np.std(runtimes)
            #range_time = max(runtimes) - min(runtimes)
            #_LOGGER.info("For submodule {}: \nforward_time average = {} \nstd dev = {} \nstd/mean ratio = {} \nrange/mean = {} \n---------".format(
            #        sub_module.__class__.__name__, mean_time ,std_time ,  std_time/mean_time, range_time/mean_time))
            
        ## Remove all the hooks that were added
        for handle in this_profiler.hook_handles:
            handle.remove()

    def run(self, criterion = None, optimizer_name = None, learning_rate = None):
        """
        :return: the model's output of the final round
        """
        _LOGGER.info("Profiling started")
        self.sub_module_nodes = {}
        self.layer_no = 0
        self.recur_function(self.model)
        
        ##################### Dummy inputs, optimizer and error criterion #################

        if self.model_type  == 'inception':
            inp_data   = torch.rand( (self.rounds,) + (self.batch_size,) + self.input_size)
            labels_data    = torch.randn( (self.rounds,) + (self.batch_size,) + (self.output_size,))
            
            if optimizer_name == None:
                optimizer = optim.SGD(self.model.parameters(), lr = 0.0001)
            if criterion == None:
                criterion = nn.MSELoss()

        elif self.model_type  == 'gnmt':
            inp_enc_data = torch.randint(self.model_info['vocab_size'], (self.rounds, self.batch_size, self.model_info['max_sequence_length']))
            inp_dec_data = torch.randint(self.model_info['vocab_size'], (self.rounds,self.batch_size, self.model_info['max_sequence_length']))
            inp_seq_len_data = torch.sort(torch.randint(self.model_info['min_sequence_length'], self.model_info['max_sequence_length'], (self.rounds,self.batch_size)), descending=True)[0]
            labels_data = torch.empty(self.rounds, self.batch_size,self.model_info['vocab_size'], dtype=torch.long).random_(2)

            if optimizer_name == None:
                optimizer = optim.SGD(self.model.parameters(), lr = 0.0001)
            if criterion == None:
                criterion = nn.CrossEntropyLoss(ignore_index=0, reduction='sum')

        elif self.model_type  == 'transformer':
            torch.random.manual_seed(0)
            inp_data = [None for _ in range(len(self.input_size))]
            for i in range(len(self.input_size)): # important to have requires_grad=True, else grad_fn may be None (in some cases)
                if self.input_size[i][0] == "int":
                    inp_data[i] = torch.randint(self.model_info['vocab_size'], (self.rounds, self.batch_size,) + self.input_size[i][1])
                else:
                    inp_data[i] = torch.randn((self.rounds,self.batch_size,) + self.input_size[i])
            labels_data = torch.randn( (self.rounds,self.batch_size,) + self.output_size)

            if optimizer_name == None and len(list(self.model.parameters()))>0:
                optimizer = optim.SGD(self.model.parameters(), lr = 0.0001)
            if criterion == None:
                criterion = nn.MSELoss()
        
        #elif --> ***For a new model add the appropriate: inp_data, labels_data, optimizer, criterion***
        else:
            raise ValueError("Invalid model name! It should be one of 'inception', 'gnmt' or 'transformer")

        if (optimizer_name is not None) and (len(list(self.model.parameters()))>0):
            optimizer = utilsf.get_optimizer(optimizer_name, self.model.parameters(), learning_rate)

        optimizer.zero_grad()
        #####################################################################
        
        last_output = None # this is the output of the final round

        for batch_idx in range(self.rounds):
            self.cur_round = batch_idx
            self.back_record = []

            if self.model_type  == 'inception':
                inp = inp_data[batch_idx].to(self.gpu)
                inp.requires_grad = True
            elif self.model_type  == 'gnmt':
                inp_enc = inp_enc_data[batch_idx].to(self.gpu)
                inp_seq_len =  inp_seq_len_data[batch_idx].to(self.gpu)
                inp_dec = inp_dec_data[batch_idx].to(self.gpu)
            elif self.model_type  == 'transformer':
                inp = [None for _ in range(len(self.input_size))]
                for i in range(len(self.input_size)):
                    inp[i] = inp_data[i][batch_idx].to(self.gpu)
                    if self.input_size[i][0] != "int":
                        inp[i].requires_grad = True
                inp = tuple(inp)
            #elif --> ***For a new model do the following:
            #         get one batch inp from inp_data, make it's requires_grad = True
            else:
                raise ValueError("Unknown model type.")

            labels = labels_data[batch_idx].to(self.gpu)
            if len(list(self.model.parameters()))>0:
                optimizer.zero_grad()

            ######### forward run ###########
            torch.cuda.synchronize(self.gpu)
            if self.model_type  == 'inception':
                output = self.model(inp)
            elif self.model_type  == 'gnmt':
                output = self.model(inp_enc, inp_seq_len, inp_dec)
            elif self.model_type  == 'transformer':
                output = self.model(*inp)
                if isinstance(output, tuple) :
                    output = output[0]
            ## elif --> ** for a new model pass the inputs defined above to self.model and 
            ##             get the output
            torch.cuda.synchronize(self.gpu)
            ######### compute loss ###########
            loss = criterion(output, labels)

            ######### backward ###########
            ## add the start time of backward 
            self.back_record.append(('start', time.time() * 1000))
            if batch_idx == self.rounds - 1:
                #loss.backward(loss, retain_graph=True)
                loss.backward(loss)
                last_output = output
            else:
                loss.backward(loss)

            if batch_idx < self.ignore_rounds:
                continue
            else:
                ## calculate the backward runtime for each layer by calculating the time differences between each timestamp
                ## Note: This is not an accurate measurement of backward time. We do not use backward times in Baechi.
                ## Currently Pytorch does not allow for a backward_pre_hook. However, if required the patch
                ## described here can be applied: https://github.com/zhuwenxi/pytorch-profiling-tool 
                for i in range(len(self.back_record) - 1, 0, -1):
                    now = self.back_record[i]
                    prev = self.back_record[i - 1]
                    cur_node = self.sub_module_nodes[now[0]]
                    cur_node.weight_backward += (now[1] - prev[1]) / (self.rounds - self.ignore_rounds)
                    self.sub_module_nodes[now[0]] = cur_node

            ### We do NOT perform step to not change the original parameters during profiling
            #if len(list(self.model.parameters()))>0: 
            #    optimizer.step()

        self.detach(self.model)
        _LOGGER.info("Profiling completed \n"+"#"*80)
        return last_output

########################################################################