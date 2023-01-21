"""Topological sort based placement."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import copy

from placer import placer_utils as utils
from placer import device as device_wrapper
from utils import logger

_LOGGER = logger.get_logger(__file__, level=logger.INFO)


class Topo():
    """Topological sort based placement.

    Places operator until the device get full."""

    def __init__(self, op_graph, device_graph):
        self.op_graph = op_graph
        self.device_graph = device_graph
        self._device_wrapper_cls = device_wrapper.DeviceWrapper
        self._devices = {
            device_id: self._device_wrapper_cls(
                device_id, self.device_graph, self.op_graph, False)
            for device_id in self.device_graph}
        topo_order_id_tuples = sorted([
            (topo_order, op_id) for op_id, topo_order
            in self.op_graph.nodes(data='id') # in m_topo_v1 -> in self.op_graph.nodes(data='topo_order')
        ])
        self._sorted_op_ids = [op_id for _, op_id in topo_order_id_tuples]

    def initialize(self):
        """Initializes."""
        for _, op_data in self.op_graph.nodes.items():
            op_data['executed_out_count'] = 0

    def is_feasible(self, op_data, device):
        """Returns whether the operator placement on the device is feasible."""
        return device.is_placeable(op_data)

    def _place_ops(self):
        current_op_index = 0
        for _, device in sorted(self._devices.items()):
            occupied_memory = 0 
            while current_op_index < len(self._sorted_op_ids):
                op_id = self._sorted_op_ids[current_op_index]
                op_data = self.op_graph.nodes[op_id]
                if not self.is_feasible(op_data, device):
                    # no more operator can be placed on this device
                    _LOGGER.info("Device {} has reached its cap".format(
                        device.id))
                    break
                device.place_op(op_id)
                self.op_graph.nodes[op_id]["topo_order"] = current_op_index
                occupied_memory += op_data['permanent_mem']
                current_op_index += 1
            _LOGGER.info('On device {} memory occupied = {} '.format(device.id, 
                    utils.humanize_num_bytes(occupied_memory)))
                
        if current_op_index != len(self._sorted_op_ids):
            raise RuntimeError(
                '{} operators cannot be placed on devices.'.format(
                    len(self._sorted_op_ids) - current_op_index))

    def run(self):
        """Places operators on devices based on the m_topo algorithm."""
        _LOGGER.info('Topo placement stats:')
        self._place_ops()


def _calculate_max_memory_per_device(op_graph, device_graph):
    required_memory = 0
    max_op_memory = 0
    for _, op_data in op_graph.nodes.items():
        op_memory = op_data['permanent_mem']
        #max_op_memory = max(max_op_memory,op_data['peak_mem'])
        max_op_memory = max(max_op_memory,op_memory)
        required_memory += op_memory

    _LOGGER.info('required memory=%s, max op memory=%s',
                 utils.humanize_num_bytes(required_memory),
                 utils.humanize_num_bytes(max_op_memory))

    max_memory_per_device = required_memory // device_graph.number_of_nodes()
    max_memory_per_device += max_op_memory
    _LOGGER.info('Max memory per device: %s',
                 utils.humanize_num_bytes(max_memory_per_device))
    # assumes that each device has memory capacity larger than
    # max_memory_per_device above...
    return max_memory_per_device


class TopoUniform(Topo):
    """Topological sort placement that places ops over devices uniformly."""

    def __init__(self, op_graph, device_graph):
        super(TopoUniform, self).__init__(op_graph, device_graph)
        self._max_memory_per_device = None

    def initialize(self):
        super(TopoUniform, self).initialize()
        self._max_memory_per_device = _calculate_max_memory_per_device(
            self.op_graph, self.device_graph)

    def is_feasible(self, op_data, device):
        _LOGGER.info("Checking for device {} for availability of {} with cap {}".format(
            device.id, utils.humanize_num_bytes(op_data['peak_mem']),
            utils.humanize_num_bytes(self._max_memory_per_device)))
        #_LOGGER.info(op_data)
        return (device.used_memory + op_data['peak_mem'] <= self._max_memory_per_device)


def m_topo(op_graph, device_graph, uniform=True):
    """Places operators on devices evenly by using the topological sort.

    Args:
        op_graph: simulation graph
        device_graph: device graph
        uniform: flag whether # ops per device are uniformly distributed
                 over devices
    """
    _LOGGER.info('m-Topo starts executing:')
    topo_cls = TopoUniform if uniform else Topo

    topo = topo_cls(copy.deepcopy(op_graph), copy.deepcopy(device_graph))
    topo.initialize()
    topo.run()

    utils.transfer_placement(topo.op_graph, op_graph)

    return op_graph


