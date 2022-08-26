import torch
import networkx as nx
import random
import time
from utils import logger

import util_functions as utilf
import config


_LOGGER = logger.get_logger(__file__, level=logger.INFO)

#############################################################################################################
### From placer_lib in baechi TF
def create_device_graph(devices):
    """Creates a placement graph for devices and network.
    Args:
        devices: device information list
    """
    graph = nx.Graph()

    for device_id, device_info in devices.items():
        graph.add_node(device_id,
                       id = device_id,
                       name = device_info["name"],
                       used_mem = 0,
                       peak_mem = 0,
                       memory_limit = device_info["memory_size"])

    for i in graph.nodes:
        for j in graph.nodes:
            # TODO: should this be added?
            if i == j:
                graph.add_edge(i, i, weight=0)
            else:
                graph.add_edge(i, j, weight=1)
    return graph

#############################################################################################################

def create_op_graph(var, cur_model, graph_info):
    """
    this function build a DiGraph for the model, by tracing the grad function of each layer's output
    :return: the DiGraph. Only the nodes from autograd graph that correspond to some layer are added. 
    Rest are shunted
    """
    _LOGGER.info("Operator graph is being created by traversing the autograd graph")
    dot = nx.DiGraph()
    seen = set()
    output_nodes = (var.grad_fn,) if not isinstance(var, tuple) else tuple(v.grad_fn for v in var)

    def add_nodes(var):
        _LOGGER.info("Variable encountered: {}".format(var))
        if var not in seen:
            cur_id = None
            if var.metadata != {}:
                if ('module' in var.metadata):
                    # this submodule has a forward function, so it's information is previously recorded in Profiling
                    cur_id = id(var.metadata['module'])
                    # retrieve the node representing this submodule
                    cur_node = cur_model.sub_module_nodes[id(var.metadata['module'])]
                    rand_perturb_factor = random.uniform(1.0 - graph_info['perturb_factor'],\
                                                         1.0 + graph_info['perturb_factor'])
                    #print("rand_perturb_factor = ", rand_perturb_factor)
                    dot.add_node(id(var.metadata['module']), 
                                 model            = str(cur_node.module), 
                                 name             = str(cur_node.name), 
                                 weight           = cur_node.weight_forward*graph_info['node_weight_factor']*rand_perturb_factor, #node_weight_factor=3 if reverse graph used #perturb_factor for sensitivity exp
                                 reverse_weight   = cur_node.weight_backward*rand_perturb_factor ,
                                 id               = id(var.metadata['module']), 
                                 topo_order       = None, 
                                 temporary_mem    = cur_node.temporary_mem,  
                                 permanent_mem    = cur_node.permanent_mem, #persistent_memory=cur_node.permanent_mem,
                                 peak_mem         = cur_node.peak_mem,
                                 parameter_mem    = cur_node.parameter_mem + cur_node.parameter_mem, #parameters and its gradient mem
                                 output_mem       = cur_node.output_mem )

                    assert cur_node.peak_mem == (cur_node.temporary_mem + cur_node.permanent_mem),\
                     "Peak memory does not match up"

                    if hasattr(var, 'next_functions'):
                        for u in var.next_functions:
                            if u[0] is not None and torch.is_tensor(u[0]) is False and hasattr(u[0], 'variable') is False:
                                if u[0].metadata != {}:
                                    if ('module' in u[0].metadata):
                                        next_id = id(u[0].metadata['module'])
                                        cur_model.sub_module_nodes[next_id].children.add(cur_id)
                                        cur_model.sub_module_nodes[cur_id].parent.add(next_id)
                                    elif ('parent' in u[0].metadata):
                                        u[0].metadata['parent'].add(cur_id)
                                    else:
                                        raise Exception("Error:", u[0], " has metadata that is neither module nor parent!")
                                        return 0
                                else:
                                    u[0].metadata['parent'] = set()
                                    u[0].metadata['parent'].add(cur_id)
                                    
                                add_nodes(u[0])
                                
                elif ('parent' in var.metadata):
                    ## these nodes are shunted
                    ## eg: SelectBackward, AddBackward0,MulBackward0,UnsqueezeBackward0, 
                    ##     SliceBackward, TBackward
                    _LOGGER.info("\t variable is shunted.")
                    cur_id_list = []
                    for parent in var.metadata['parent']:
                        cur_id_list.append(parent)
                    if hasattr(var, 'next_functions'):
                        for u in var.next_functions:
                            if u[0] is not None and torch.is_tensor(u[0]) is False and hasattr(u[0], 'variable') is False:
                                if u[0].metadata != {}:
                                    if ('module' in u[0].metadata):
                                        next_id = id(u[0].metadata['module'])
                                        for cur_id in cur_id_list:
                                            cur_model.sub_module_nodes[next_id].children.add(cur_id)
                                            cur_model.sub_module_nodes[cur_id].parent.add(next_id)
                                    elif ('parent' in u[0].metadata):
                                        for cur_id in cur_id_list:
                                            u[0].metadata['parent'].add(cur_id)
                                    else:
                                        raise Exception("Error:", u[0], " has metadata that is neither module nor parent!")
                                        return 0
                                else:
                                    u[0].metadata['parent'] = set()
                                    for cur_id in cur_id_list:
                                        u[0].metadata['parent'].add(cur_id)
                                add_nodes(u[0])
                
            else:
                ## All functions will have either 'module' or 'parent' metadata
                _LOGGER.info(var.__dict__)
                raise Exception("Error:", var, " does not have any metadata!")
                return 0

            seen.add(var)

    if isinstance(var, tuple):
        # handle multiple outputs
        for v in var:
            add_nodes(v.grad_fn)
    else:
        add_nodes(var.grad_fn)
    
    return dot

###################################################################################################################
def build_graph(final_output, profiled_model, graph_info):
    """
    this is the main function to call for building the graph, it calls profiling and make dot, and made further improvements
    :param model: input model
    :param gpu: which gpu to place the profiler in
    :param rounds: number of rounds to run the profiling
    :return: the DiGraph, and the Profiling object
    """
    start_graph_time = time.time()
    return_graph = create_op_graph(final_output, profiled_model, graph_info)
    end_graph_time = time.time()
    _LOGGER.info("Graph creation time =  {} sec \n ************".format((end_graph_time - start_graph_time) ))
    
    utilf.topological_sort(profiled_model)
    
    # use the sudo id instead of hash_id, this is for scheduler purpose
    _LOGGER.debug("Node id's in operator graph are changed to topo order from profiler")
    for node_id in profiled_model.sub_module_nodes.keys():
        model_node = profiled_model.sub_module_nodes[node_id]
        assert not(len(model_node.parent)==0 and len(model_node.children)==0), "Module defined but not used"
        graph_node = return_graph.nodes[node_id]
        graph_node["id"] = model_node.id
        graph_node["topo_order"] = None
        return_graph.add_nodes_from([(model_node.id, graph_node)])
        return_graph.remove_node(node_id)

    # change the id of edges
    _LOGGER.info("Filling Edges of the graph")
    edge_count = 0
    for node in profiled_model.sub_module_nodes.keys():
        children = profiled_model.sub_module_nodes[node].children
        node_new_id = profiled_model.sub_module_nodes[node].id
        for kid in children:
            rand_perturb_factor = random.uniform(1.0 - graph_info['perturb_factor'],\
                                                 1.0 + graph_info['perturb_factor'])
            #print("rand_perturb_factor_comm = ", rand_perturb_factor)
            kid_new_id = profiled_model.sub_module_nodes[kid].id
            #  For now, we assume one edge is one tensor, carrying all of from_node's outputs
            edge_data = {
                "weight": 0, "id": edge_count, "tensor": [], 
                "requested_bytes": profiled_model.sub_module_nodes[node].output_mem
            }
            if edge_data['requested_bytes'] != 0:
                edge_data['weight'] = (graph_info['m_comm_time_vs_data_size']*edge_data['requested_bytes'] +\
                                       graph_info['b_comm_time_vs_data_size'] ) * rand_perturb_factor
            
            edge_data['tensor'] = { "name" : str(edge_count), "transfer_time" : edge_data['weight'] , 
                                    "bytes" : edge_data['requested_bytes'] }

            if config.REVERSE_GRAPH_FLAG:
                return_graph.add_edge(kid_new_id, node_new_id, **edge_data) #reverse graph
            else:
                return_graph.add_edge(node_new_id, kid_new_id, **edge_data)
                
            edge_count += 1

    return return_graph, profiled_model


