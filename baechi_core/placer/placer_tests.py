import sys, os
sys.path.append("../..")
sys.path.append("..")

import networkx as nx
import m_etf
import m_sct_test as m_sct
import m_topo
import dummy_graphs as dg


comm_factor    = 0

def create_device_graph(no_devices, device_max_mem):
    graph = nx.Graph()

    for device_id in range(no_devices):
        graph.add_node(device_id,
                       id = device_id,
                       name = str(device_id),
                       used_mem = 0, 
                       peak_mem = 0,
                       memory_limit = device_max_mem[device_id])

    for i in graph.nodes:
        for j in graph.nodes:
            # TODO: should this be added?
            if i == j:
                graph.add_edge(i, i, weight=0)
            else:
                graph.add_edge(i, j, weight=1)
    return graph


def create_op_graph(graph_info):
    graph = nx.DiGraph()
    edge_count = 0
    for node in graph_info:
        graph.add_node(node, 
                id            = graph_info[node]['id'],
                name          = graph_info[node]['name'],
                weight        = graph_info[node]['forward_time'],
                topo_order    = None, 
                temporary_mem = graph_info[node]['temporary_mem'],  
                permanent_mem = graph_info[node]['permanent_mem'],
                peak_mem      = graph_info[node]['permanent_mem'] + graph_info[node]['temporary_mem'],
                parameter_mem = graph_info[node]['permanent_mem']/2,)
        child_no = 0
        for child in graph_info[node]['children' ]:
            edge_data = {"weight": graph_info[node]['edge_weight'][child_no], 
                         "id"    : edge_count, 
                         "tensor": [], 
                         "requested_bytes": comm_factor*graph_info[node]['edge_weight'][child_no] }
            edge_data['tensor'] = { "name" : str(edge_count), 
                                    "transfer_time" : edge_data['weight'] , 
                                    "bytes" : edge_data['requested_bytes'] }
            graph.add_edge(node, child, **edge_data)
            edge_count = edge_count + 1
            child_no   = child_no + 1
    return graph


#graph_info = dg.graph_fig1
#no_devices     = 2

graph_info = dg.graph_fig21
no_devices     = 2

#graph_info = dg.graph_fig3
#no_devices     = 3

device_max_mem = [4.0 for _ in range(no_devices)]

for node in graph_info:
    #graph_info[node]['name'] = 'op'+str(node+1)
    graph_info[node]['name'] = 'op'+str(node)

op_graph  = create_op_graph(graph_info)
dev_graph = create_device_graph(no_devices, device_max_mem)
#placed_op_graph = m_sct.m_sct(op_graph, dev_graph)
placed_op_graph = m_etf.m_etf(op_graph, dev_graph)


for node, dev in placed_op_graph.nodes.data('p'):
    print("node:", graph_info[node]['name'])
    print("Device:", dev)
    print("-"*20)

