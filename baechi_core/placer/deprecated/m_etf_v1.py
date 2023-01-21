"""Earliest Time First Placement."""
# pylint: disable=invalid-name
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import copy
from enum import Enum

from placer import device as device_wrapper
from placer import placer_utils
from utils import logger

_LOGGER = logger.get_logger(__file__, level=logger.INFO)


def get_ready_ts(op_graph, target_op_id, device_id):
    """Returns a ready timestamp for the given operator on the device.

    DEPRECATED

    CAVEATS: this function does not consider device data transfer channels.
    """
    in_edges = list(op_graph.in_edges(target_op_id, data=True))
    ready_ts = 0
    for in_edge in in_edges:
        from_op_id, _, edge_data = in_edge
        from_op = op_graph.nodes[from_op_id]
        data_transfer_time = (
            0 if from_op["p"] == device_id else edge_data["weight"])
        ready_ts = max(ready_ts, from_op["end_ts"] + data_transfer_time)
    return ready_ts


def populate_ready_tss(op_graph, device_graph, op):
    """Populate ready/urgent timestamps for op on devices in device_graph.

    Args:
        op_graph: placement graph
        device_graph: device_graph
        op: target op node in the placement graph
    Returns:
        ready timestamps dict for the given op
    """
    op_id = op['id']
    ready_tss = {
        device_id: get_ready_ts(op_graph, op_id, device_id)
        for device_id, device_data in device_graph.nodes.items()
        if op['memory'] <= device_data['memory_limit'] - device_data['size']}
    op['ready_tss'] = ready_tss
    # TODO: urgent_ts might change on runtime due to the memory constraint.
    #       fix this.
    op['urgent_ts'] = max(ready_tss.values())

    if len(ready_tss) == 0:
            # no device can execute this op
        raise RuntimeError(
            'Op id=%d cannot run on any available device' % op_id)

    return ready_tss


class DeviceState(Enum):
    """Device state."""
    FREE = 0
    BUSY = 1
    AWAKE = 2  # used in sct


class DeviceWrapper(device_wrapper.DeviceWrapper):
    """DeviceWrapper for ETF."""

    def __init__(self, device_id, device_graph, op_graph):
        super(DeviceWrapper, self).__init__(device_id, device_graph, op_graph)
        self._current_ts = 0
        self._state = DeviceState.FREE
        self._last_op = None
        self._ready_ts_ops = placer_utils.SortedTimestampOps()

    def get_next_ts(self):
        """Returns the timestamp when this device can have any action."""
        next_ts = None
        if self._state == DeviceState.BUSY:
            next_ts = self.next_available_ts
        elif self._state == DeviceState.FREE:
            if len(self._ready_ts_ops) > 0:
                next_ts = self._ready_ts_ops[0].ts
        else:
            raise ValueError('Unknown state: {}'.format(self._state))
        if next_ts is not None and next_ts <= self._current_ts:
            raise ValueError('Timestamp should move forward')
        return next_ts

    def _remove_not_executable_ops(self):
        op_ids_to_remove = []
        new_ready_ts_ops = []
        for ready_ts_op in self._ready_ts_ops:
            if ready_ts_op.op['memory'] > self.available_memory:
                op_ids_to_remove.append(ready_ts_op.op_id)
            else:
                new_ready_ts_ops.append(ready_ts_op)
        self._ready_ts_ops = placer_utils.SortedTimestampOps(new_ready_ts_ops)

        for op_id in op_ids_to_remove:
            # remove this device from ready_tss in order to change state
            # to AWAKE correctly in advance().
            op = self._op_graph.nodes[op_id]
            del op['ready_tss'][self.id]

    def _get_earlist_ready_op(self):
        if len(self._ready_ts_ops) > 0:
            first_ready_ts_op = self._ready_ts_ops[0]
            if first_ready_ts_op.ts <= self._current_ts:
                first_ready_ts_op.op['ready_ts'] = first_ready_ts_op.ts
                return first_ready_ts_op.op

        return None

    def _schedule_op(self, op):
        op_id = op['id']
        self.place_op(op_id)
        self.run_op(op_id)
        self._remove_not_executable_ops()
        self._state = DeviceState.BUSY
        self._last_op = op
        # this op should be removed by calling remove_ready_op() later

    def _do_free_action(self):
        ready_op = self._get_earlist_ready_op()
        if ready_op:
            self._schedule_op(ready_op)
            return ready_op

        return None

    def advance(self, ts):
        """Advances the device time to the given timestamp."""
        self._current_ts = ts
        new_ready_ops = []
        if self._current_ts == self.next_available_ts:
            if self._last_op is not None:
                new_ready_ops = placer_utils.process_finished_op(
                    self._op_graph, self._last_op['id'])
                for new_ready_op in new_ready_ops:
                    populate_ready_tss(
                        self._op_graph, self._device_graph, new_ready_op)
        return new_ready_ops

    def change_state(self):
        """Changes the device state."""
        if self._current_ts < self.next_available_ts:
            self._state = DeviceState.BUSY
        else:
            self._state = DeviceState.FREE

    def schedule(self):
        """Schedules a new operator if available.

        Returns:
            a operator that is scheduled.
        """
        self.change_state()
        if self._state == DeviceState.FREE:
            return self._do_free_action()

        assert self._state == DeviceState.BUSY
        return None

    def add_ready_op(self, op):
        """Adds the given operator to the ready queue.

        Returns:
            True if succeeded. False, otherwise.
        """
        if op['memory'] <= self.available_memory:
            self._ready_ts_ops.add_op(op['ready_tss'][self.id], op)
            return True

        op['ready_tss'].pop(self.id)
        return False

    def remove_ready_op(self, op):
        """Removes the given operator from the ready queue."""
        self._ready_ts_ops.remove_op(op)


class ETF():
    """Earliest Time First Placement."""

    def __init__(self, op_graph, device_graph):
        self.op_graph = op_graph
        self.device_graph = device_graph
        self.devices = [
            DeviceWrapper(device_id, self.device_graph, self.op_graph)
            for device_id in device_graph.nodes]

        device = self.devices[0]
        _LOGGER.info('DeviceWrapper type: %s', device.type())
        self._get_op_memory_fn = device.get_op_memory

        self._num_scheduled_op = 0

    def _initialize_ops(self):
        initialized_ops = []
        for op_id, op_data in self.op_graph.nodes.items():
            op_data['ready_count'] = 0
            op_data['memory'] = self._get_op_memory_fn(op_data)
            if self.op_graph.in_degree(op_id) == 0:
                ready_tss = populate_ready_tss(
                    self.op_graph, self.device_graph, op_data)
                for device_id in ready_tss.keys():
                    self.devices[device_id].add_ready_op(op_data)
                initialized_ops.append(op_data)
        return initialized_ops

    def _process_scheduled_op(self, scheduled_op):
        if scheduled_op is not None:
            # remove the scheduled op from all devices
            for device in self.devices:
                device.remove_ready_op(scheduled_op)
            self._num_scheduled_op += 1

    def _process_new_ready_ops(self, new_ready_ops):
        for new_ready_op in new_ready_ops:
            for device_id in new_ready_op['ready_tss'].keys():
                self.devices[device_id].add_ready_op(new_ready_op)

    def _run_schedule_step(self, ts):
        """Runs a single schedule step at the given timestamp.

        Args:
            ts: timestamp.
        Returns:
            next schedule step time.
        """
        new_ready_ops = []
        for device in self.devices:
            new_ready_ops.extend(device.advance(ts))

        self._process_new_ready_ops(new_ready_ops)

        # schedule new ops
        for device in self.devices:
            self._process_scheduled_op(device.schedule())

        min_next_ts = None
        for device in self.devices:
            next_ts = device.get_next_ts()
            if next_ts is not None:
                min_next_ts = min(min_next_ts or next_ts, next_ts)

        return min_next_ts

    def _run_schedule(self):
        # start to schedule and place ops
        self._initialize_ops()

        ts = 0  # current time
        while True:
            next_ts = self._run_schedule_step(ts)
            if next_ts is None:
                break
            ts = next_ts

        assert self._num_scheduled_op == self.op_graph.number_of_nodes(), \
            "# scheduled ops={}, # ops={}".format(
                self._num_scheduled_op, self.op_graph.number_of_nodes())
        return ts

    def run(self):
        """Runs the placement."""
        runtime = self._run_schedule()
        _LOGGER.info('ETF estimated runtime: %f', runtime / 1e6)
        return runtime


def m_etf(op_graph, device_graph):
    """Runs m_etf placement."""
    etf = ETF(copy.deepcopy(op_graph), copy.deepcopy(device_graph))
    etf.run()
    placer_utils.transfer_placement(etf.op_graph, op_graph)
    return op_graph
