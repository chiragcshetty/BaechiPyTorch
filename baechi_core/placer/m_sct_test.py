"""Memory-constrainted shortest communication time placement."""
from __future__ import absolute_import, division, print_function

import copy

from placer import placer_utils
from placer.deprecated.m_sct_v1 import FavoriteChildLPSolver
from placer.m_etf import ETF, DeviceState, ETFDevice
from utils import logger

_LOGGER = logger.get_logger(__file__, level=logger.INFO)


class SCTDevice(ETFDevice):
    """SCT Device wrapper."""

    def _get_earliest_urgent_op(self, ready_op_manager):
        # assumes that 'schedule_tss' is updated at op_data
        urgent_ts_ops = placer_utils.SortedTimestampOps()
        for ready_op in ready_op_manager:
            if 'p' in ready_op and ready_op['p'] != self.id:
                # this op is already placed on another device.
                continue
            if self.is_placeable(ready_op):
                urgent_ts_ops.add_op(ready_op['urgent_ts'], ready_op)

        # TODO: If there are multiple ops that have the same ready/urgent tss,
        #       pick the one whose parent's device is not this op.
        #       This is for respecting favorite child. (same for ready_op)
        return urgent_ts_ops[0].op if len(urgent_ts_ops) > 0 else None

    def _get_op_on_await(self, ready_op_manager):
        urgent_op = self._get_earliest_urgent_op(ready_op_manager)
        if urgent_op and urgent_op['urgent_ts'] <= self._current_ts:
            return urgent_op

        # the favorite child is not executed yet.
        # wait until the last op's favorite child is ready to run
        fc_op = self._op_graph.nodes[self._last_op['favorite']]
        if self.get_schedule_ts(fc_op) <= self._current_ts:
            return fc_op

        return None

    def get_op_to_schedule(self, ready_op_manager):
        """Returns an operator to schedule if available.

        Returns:
            a operator that is scheduled.
        """
        # assumes schedule_tss in op_data is updated.
        if self._current_ts < self.next_available_ts:
            self._state = DeviceState.BUSY
            return None

        # device is free
        self._state = DeviceState.FREE
        if self._last_op is not None:
            # check whether there is a favorite child op that this device
            # can run earlist than other devices.
            # if so, set the state to AWAKE.
            fc_op_id = self._last_op['favorite']
            if fc_op_id != -1:
                fc_op = self._op_graph.nodes[fc_op_id]
                if 'p' not in fc_op and 'schedule_tss' in fc_op:
                    # the favorite child does not run yet but is ready to run
                    fc_op_schedule_tss = fc_op['schedule_tss']
                    if self.id in fc_op_schedule_tss:
                        # current device can run fc_op
                        min_schedule_ts = min(fc_op_schedule_tss.values())
                        if fc_op_schedule_tss[self.id] == min_schedule_ts:
                            self._state = DeviceState.AWAKE
                            return self._get_op_on_await(ready_op_manager)

        return self._get_op_on_free(ready_op_manager)

    def get_next_ts(self, ready_op_manager):
        """Returns the timestamp when this device can have any action."""
        # assumes schedule_tss in op_data is updated.
        next_ts = None
        if self._state == DeviceState.AWAKE:
            # min(favorite child ready time, min. urgent ts)
            fc_op = self._op_graph.nodes[self._last_op['favorite']]
            fc_op_schedule_tss = fc_op['schedule_tss']
            if 'p' not in fc_op and self.id in fc_op_schedule_tss:
                next_ts = fc_op_schedule_tss[self.id]

            earliest_urgent_op = self._get_earliest_urgent_op(ready_op_manager)
            if earliest_urgent_op is not None:
                next_urgent_ts = earliest_urgent_op['urgent_ts']
                next_ts = min(next_ts or next_urgent_ts, next_urgent_ts)
            if next_ts is not None and next_ts <= self._current_ts:
                raise ValueError('Timestamp should move forward')
            return next_ts

        return super(SCTDevice, self).get_next_ts(ready_op_manager)


class SCT(ETF):
    """Memory-constrainted shortest communication time placement."""

    def __init__(self, op_graph, device_graph,
                 log_file=None, threshold=0.5):
        # pylint: disable=too-many-arguments
        super(SCT, self).__init__(op_graph, device_graph, log_file)
        self._favorite_child_lp_solver = FavoriteChildLPSolver(
            op_graph, threshold)

    def _assign_favorite_child(self, favorite_child_edges):
        for _, op_data in self.op_graph.nodes.items():
            op_data['favorite'] = -1
            op_data['parent'] = -1

        num_favorite_child = 0
        num_favorite_child_change = 0
        for op1_id, op2_id, edge_id in self.op_graph.edges(data='id'):
            if favorite_child_edges[edge_id] == 0:
                op1 = self.op_graph.nodes[op1_id]
                if op1['favorite'] != -1:
                    _LOGGER.debug(
                        'Changing favorite child of op %d from %d to %d',
                        op1_id,
                        op1['favorite'],
                        op2_id)
                    num_favorite_child_change += 1

                op1['favorite'] = op2_id
                _LOGGER.info('***** Fav child of %d is %d', op1_id, op2_id )
                num_favorite_child += 1

                op2 = self.op_graph.nodes[op2_id]
                op2_parent_id = op2['parent']
                if op2_parent_id != -1:
                    op2_parent = self.op_graph.nodes[op2_parent_id]
                    _LOGGER.debug(
                        'Changing favorite child of op %d from %d to none',
                        op2_parent_id,
                        op2_parent['favorite'])
                    op2_parent['favorite'] = -1
                    num_favorite_child_change += 1
                op2['parent'] = op1_id

        _LOGGER.info('# favorite child: %d', num_favorite_child)
        _LOGGER.info('# favorite child changes: %d', num_favorite_child_change)

    def initialize(self):
        """Initializes."""
        self._num_scheduled_ops = 0
        self._devices = {
            device_id: SCTDevice(device_id, self._device_graph,
                                 self.op_graph)
            for device_id in self._device_graph.nodes}
        self._ready_op_manager = placer_utils.ReadyOpManager(
            self.op_graph, self._devices, self._log_file)

        for op_id, op_data in self.op_graph.nodes.items():
            op_data['ready_count'] = 0
            if self.op_graph.in_degree(op_id) == 0:
                self._ready_op_manager.add(op_data)

        favorite_child_edges = self._favorite_child_lp_solver.run()
        self._assign_favorite_child(favorite_child_edges)


def m_sct(op_graph, device_graph, threshold=0.5, colocation=False):
    """Places operators over the devices by using SCT."""
    sct = SCT(copy.deepcopy(op_graph), copy.deepcopy(device_graph),
              threshold=threshold)
    runtime = sct.run()
    _LOGGER.info('SCT estimated runtime: %f', runtime / 1e6)
    placer_utils.transfer_placement(sct.op_graph, op_graph)
    return op_graph
