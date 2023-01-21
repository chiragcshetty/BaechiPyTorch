"""Placer utility module."""
# pylint: disable=invalid-name
from __future__ import absolute_import, division, print_function

import collections
import concurrent.futures
import functools
import operator
import sys
import os

# import matplotlib.pyplot as plt
import networkx as nx

from utils import logger

_LOGGER = logger.get_logger(__file__, level=logger.INFO)

#################################### Placement Manager #############################################
class TimestampOpTuple():
    """A tuple of timestamp and operator. Used for sorting ops according ts."""

    def __init__(self, ts, op):
        self.ts = ts
        self.op = op

    @property
    def op_id(self):
        """Returns the operator id."""
        return self.op['id']

    def __lt__(self, other):
        return ((self.ts, self.op['topo_order']) <
                (other.ts, other.op['topo_order']))

    def __eq__(self, other):
        return self.ts == other.ts and self.op == other.op

    def __repr__(self):
        return 'TimestampOpTuple(ts={}, op={})'.format(self.ts, self.op)

#------------------------------------------------------------------------------------
class SortedTimestampOps():
    """Sorted list of timestamp and operator tuples."""

    def __init__(self, *args):
        self._list = list(*args)
        self._sorted = False

    def add_op(self, ts, op):
        """Adds a new op with the given timestamp."""
        self._list.append(TimestampOpTuple(ts, op))
        self._sorted = False

    def __len__(self):
        return len(self._list)

    def remove_op(self, op):
        """Removes the given operator from the list.

        Returns:
            True if the given operator is removed. False, otherwise.
        """
        op_id = op['id']
        for i, ts_op in enumerate(self._list):
            if ts_op.op_id == op_id:
                del self._list[i]
                return True
        return False

    def _sort(self):
        if not self._sorted:
            self._list = sorted(self._list)
            self._sorted = True

    def __getitem__(self, index):
        self._sort()
        return self._list[index]

    def pop(self, index=0):
        """Pops the item at the given index."""
        self._sort()
        return self._list.pop(index)

    def __iter__(self):
        self._sort()
        return self._list.__iter__()

def find_index_of_ts_op_tuple(ts_op_tuples, op_id):
    """Finds the index of the given operator id at the ts and op tuple list."""
    for i, ts_op_tuple in enumerate(ts_op_tuples):
        if ts_op_tuple.op_id == op_id:
            return i
    return -1

#------------------------------------------------------------------------------------
def process_finished_op(op_graph, op_id):
    """Processes the finished op and returns next ops to be ready to execute.
    """
    ready_ops = []
    for _, next_op_id in op_graph.out_edges(op_id):
        next_op = op_graph.nodes[next_op_id]
        next_op['ready_count'] += 1
        in_degree = op_graph.in_degree(next_op_id)
        if next_op['ready_count'] == in_degree:
            # ready to run
            ready_ops.append(next_op)
    return ready_ops

#------------------------------------------------------------------------------------
class ReadyOpManager():
    """Ready operator manager."""

    def __init__(self, op_graph, devices, log_file=None):
        self._op_graph = op_graph
        self._devices = devices
        self._log_file = log_file
        self._ready_ops = []

    def _estimate_tensor_transfer_end_ts(
            self, tensor_data, send_device, recv_device,
            send_op, recv_op):
        """Calculates the estimated tensor transfer end timestamp."""
        end_ts = -1
        cached_tensor = recv_device.get_cached_tensor(tensor_data['name'])
        if cached_tensor is not None: # Tensor has already been transferred
            end_ts = cached_tensor['recv_end_ts']    
        else:
            send_start_ts, _ = send_device.send_tensor(tensor_data, send_op["end_ts"], actually = False)
            _, recv_end_ts = recv_device.recv_tensor(tensor_data, send_start_ts, actually = False)
            end_ts = recv_end_ts
        
        if self._log_file:
            self._log_file.write(generate_memcpy_log(
                            tensor_data, send_op, recv_op,
                            send_device.id, recv_device.id, True))

        assert end_ts > 0
        return end_ts


    def get_ready_ts(self, op_data, device):
        """Returns a ready timestamp for the operator on the device."""
        if not device.is_placeable(op_data):
            return None

        if device.id in op_data['ready_tss']:
            return op_data['ready_tss'][device.id]

        ready_ts = 0
        in_edges = list(self._op_graph.in_edges(op_data['id'], data=True))
        for in_edge in in_edges:
            from_op_id, _, edge_data = in_edge
            from_op = self._op_graph.nodes[from_op_id]
            from_op_device = self._devices[from_op['p']]
            if from_op_device.id == device.id:
                new_ready_ts = from_op["end_ts"]
            else:
                new_ready_ts = self._estimate_tensor_transfer_end_ts(
                    edge_data['tensor'],
                    send_device=from_op_device, recv_device=device,
                    send_op=from_op, recv_op=op_data)
            ready_ts = max(ready_ts, new_ready_ts)

        op_data['ready_tss'][device.id] = ready_ts
        return ready_ts

    def get_schedule_ts(self, op_data, device):
        """Returns a schedule-able timestamp for the operator on the device."""
        if 'p' in op_data and op_data['p'] != device.id:
            # get schedule-able ts only for the assigned device.
            return None
        ready_ts = self.get_ready_ts(op_data, device)
        if ready_ts is not None:
            return max(ready_ts, device.next_available_ts)
        return None

    def get_schedule_tss(self, op_data):
        """Return a schedule-able ts dict for the operator on all devices."""
        retval = {}
        for device in self._devices.values():
            schedule_ts = self.get_schedule_ts(op_data, device)
            if schedule_ts is not None:
                retval[device.id] = schedule_ts
        return retval

    def _populate_schedule_tss(self, ready_op):
        schedule_tss = self.get_schedule_tss(ready_op)
        ready_op['schedule_tss'] = schedule_tss
        ready_op['urgent_ts'] = (max(schedule_tss.values())
                                 if schedule_tss else sys.maxsize)   # used in m-sct

    def populate_schedule_tss(self):
        """Populates schedule_tss and urgent_ts of all operators."""
        for ready_op in self._ready_ops:
            self._populate_schedule_tss(ready_op)

    @staticmethod
    def _initialize_op(op_data):
        op_data['ready_tss'] = {}

    def add(self, op_data):
        """Adds the operator into the ready operator list."""
        self._initialize_op(op_data)
        return self._ready_ops.append(op_data)

    def remove(self, op_data):
        """Removes the operator from the ready operator list."""
        return self._ready_ops.remove(op_data)

    def extend(self, iterable):
        """Extends the ready operator list with the given iterable object."""
        for op_data in iterable:
            self._initialize_op(op_data)
        return self._ready_ops.extend(iterable)

    def __len__(self):
        return len(self._ready_ops)

    def __iter__(self):
        return iter(self._ready_ops)

################################################### Utilities ############################################
def humanize_num_bytes(num_bytes):
    """Returns a number of bytes string."""
    suffixes = ['B', 'KB', 'MB', 'GB', 'TB', 'PB']
    i = 0
    while num_bytes >= 1024 and i < len(suffixes)-1:
        num_bytes /= 1024.
        i += 1
    number_str = ('%.2f' % num_bytes).rstrip('0').rstrip('.')
    return '%s%s' % (number_str, suffixes[i])


def save_placement_graph(op_graph,
                         filename,
                         figsize=None,
                         font_size=12,
                         with_colocation_group=False):
    """Save placement graph to a file."""
    fig = plt.figure(figsize=figsize)
    # pos = nx.drawing.nx_agraph.graphviz_layout(op_graph)
    # pos = nx.drawing.layout.spring_layout(op_graph)
    pos = nx.drawing.layout.circular_layout(op_graph)
    # pos = nx.drawing.layout.shell_layout(op_graph)

    labels = {}
    for op_id, data in op_graph.nodes(True):
        label = ('\n'.join([data['name'], data['colocation_group']])
                 if with_colocation_group else data['name'])
        labels[op_id] = label

    nx.draw_networkx_labels(op_graph, pos, labels=labels, font_size=font_size)
    nx.draw_networkx_nodes(op_graph, pos, node_color='b')
    nx.draw_networkx_edges(op_graph, pos)
    fig.savefig(filename)

def generate_op_run_log(op_data):
    """Returns operator execution log."""
    return 'device id={} op={}, ready={}, start={}, end={}\n'.format(
        op_data['p'], op_data['name'], op_data['ready_ts'],
        op_data['start_ts'], op_data['end_ts'])


def generate_memcpy_log(
        tensor_data, from_op, to_op, from_device_id, to_device_id, cached):
    """Generates a memcpy log."""
    # pylint: disable=too-many-arguments
    return ('memcpy tensor={}, device {}->{}, cached={}, from={}, to={}, '
            'send_start={}, recv_start={}, recv_end={}, cost={}\n'.format(
                tensor_data['name'], from_device_id, to_device_id, cached,
                from_op['name'], to_op['name'], tensor_data['send_start_ts'],
                tensor_data['recv_start_ts'], tensor_data['recv_end_ts'],
                tensor_data['weight']))


def transfer_placement(from_op_graph, to_op_graph):
    """Transfers the device placement between placement graphs."""
    for op_id, p in from_op_graph.nodes.data('p'):
        to_op_graph.nodes[op_id]['p'] = p
    ### To capture the topo order
    for op_id, topo_order in from_op_graph.nodes.data('topo_order'):
        to_op_graph.nodes[op_id]['topo_order'] = topo_order

