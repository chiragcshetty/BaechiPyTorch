import logging
import operator
import math
import networkx

from utils import logger

_LOGGER = logger.get_logger(__file__)


def m_topo(op_graph, device_graph):
    """Places operators on devices evenly by using the topological sort.

    Args:
        op_graph: simualtion graph
        device_graph: device graph
    """
    sorted_ops = list(networkx.topological_sort(op_graph))

    num_ops = op_graph.number_of_nodes()
    num_ops_per_device = int(math.ceil(float(num_ops) / len(device_graph)))
    _LOGGER.info('# ops per device: %d', num_ops_per_device)

    begin = 0
    end = 0
    for device_id, device_data in device_graph.nodes(data=True):
        current_memory = 0
        memory_limit = device_data['memory_limit']
        _LOGGER.info('Memory limit of device %d: %d', device_id, memory_limit)
        end = min(num_ops, num_ops_per_device * (device_id + 1))
        # find end index by which operators can be assigned to a device.
        for i in range(begin, end):
            current_memory += op_graph.nodes[sorted_ops[i]]['memory']
            if current_memory > memory_limit:
                end = i
                break
        for i in range(begin, end):
            op_id = sorted_ops[i]
            op_graph.nodes[op_id]['p'] = device_id
        begin = end

    # initialize the ready time for each operator
    for op_id in sorted_ops:
        if op_graph.in_degree(op_id) == 0:
            op_graph.nodes[op_id]['s'] = 0.0
        # else:
            # op_graph.nodes[op_id]['s'] = float('inf')

    # initialize device timestamps
    for device_id, device_data in device_graph.nodes(data=True):
        device_data['t'] = 0.0

    # estimate makespan according to the assignment
    ops_dict = {}  # key: op_id, value: ready timestamp
    for op_id in sorted_ops:
        op_node = op_graph.nodes[op_id]
        pred_op_ids = list(op_graph.predecessors(op_id))
        ready_ts = 0
        for pred_id in pred_op_ids:
            pred_op_node = op_graph.nodes[pred_id]
            pred_op_end_time = pred_op_node['s'] + pred_op_node['weight']
            if op_node['p'] == pred_op_node['p']:
                # Ops are assigned to the same device
                ready_ts = max(ready_ts, pred_op_end_time)
            else:
                # Ops are assigned to different devices.
                # A tensor needs to be transferred.
                ready_ts = max(
                    ready_ts,
                    pred_op_end_time + op_graph[pred_id][op_id]['weight'])
        op_node['s'] = ready_ts
        ops_dict[op_id] = ready_ts

    # sort ops by their ready timestamps
    ops_dict = sorted(ops_dict.items(), key=operator.itemgetter(1))

    makespan = 0
    for op_id, ready_ts in ops_dict:
        op_node = op_graph.nodes[op_id]
        device_node = device_graph.nodes[op_node['p']]
        op_node['s'] = max(ready_ts, device_node['t'])
        device_node['t'] = op_node['s'] + op_node['weight']
        makespan = max(makespan, device_node['t'])

    return op_graph, makespan
