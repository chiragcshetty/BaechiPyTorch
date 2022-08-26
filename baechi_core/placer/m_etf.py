"""Memory-constrainted earliest time first placement."""
from __future__ import absolute_import, division, print_function

import copy
from collections import namedtuple
from enum import Enum

from placer import device as device_wrapper
from placer import placer_utils
from utils import logger

_LOGGER = logger.get_logger(__file__, level=logger.INFO)


class DeviceState(Enum):
    """Device state."""
    FREE = 0
    BUSY = 1
    AWAKE = 2  # used in SCT

################################################################################################################
# ETFDevice = wrapper for a device D
# Attributes = [current_ts, state, peak_mem, last op, next_available_ts(from DeviceWrapper)]
# _memory_check = flag indicates if memory limit must be checked (True for memory constrained 
#                 ETF, False for simple ETF)
#
# --------  Methods (all schedule related. memory is handled in device_wrapper.DeviceWrapper) -------------------
# advance(ts)             = Advances the device time to ts. If ts > next_available_ts then it simulates finishing
#                          the last_op on D and returns a list of ops that can be scheduled on D ts onwards.
# get_schedule_ts(op)     = Returns ts at which the op can be scheduled on D.
# _get_earlist_ready_op() = Returns the op that can be scheduled earliest on D
# _get_op_on_free()       = Return op, if any op can be scheduled on D at current_ts, 
# get_op_to_schedule()    = Sets the state of D (BUSY/FREE) and if FREE, returns _get_op_on_free()
# get_next_ts()           = Returns ts when D becomes free and if free, returns when any op becomes 
#                           placebale on D next
##################################################################################################################


class ETFDevice(device_wrapper.DeviceWrapper):
    """SCT Device wrapper."""

    def __init__(self, device_id, device_graph, op_graph):
        super(ETFDevice, self).__init__(device_id, device_graph, op_graph)
        self._current_ts = 0
        self._state = DeviceState.FREE
        self._last_op = None

    #-------------------- Scheduling functions -------------------- 
    def advance(self, timestamp):
        """Advances the device time to the given timestamp."""
        self._current_ts = timestamp
        new_ready_ops = []
        if self._current_ts == self.next_available_ts:
            if self._last_op is not None:
                new_ready_ops = placer_utils.process_finished_op(
                    self._op_graph, self._last_op['id'])
        if self.next_available_ts < self._current_ts:
            self._next_available_ts = self._current_ts
        return new_ready_ops

    def get_schedule_ts(self, op_data):
        """Returns schedule-able timestamp for the operator."""
        schedule_tss = op_data['schedule_tss']
        if self.id in schedule_tss:
            return schedule_tss[self.id]
        return None

    def _get_earlist_ready_op(self, ready_op_manager):
        # assumes that 'schedule_tss' is updated at op_data
        ready_ts_ops = placer_utils.SortedTimestampOps()
        for ready_op in ready_op_manager:
            schedule_ts = self.get_schedule_ts(ready_op)
            if schedule_ts is not None:
                ready_ts_ops.add_op(schedule_ts, ready_op)

        return ready_ts_ops[0].op if len(ready_ts_ops) > 0 else None

    def _get_op_on_free(self, ready_op_manager):
        ready_op = self._get_earlist_ready_op(ready_op_manager)
        if ready_op and self.get_schedule_ts(ready_op) <= self._current_ts:
            return ready_op
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
        return self._get_op_on_free(ready_op_manager)

    def get_next_ts(self, ready_op_manager):
        """Returns the timestamp when this device can have any action."""
        # schedule_tss should be populated.
        next_ts = None
        if self._state == DeviceState.BUSY:
            next_ts = self.next_available_ts
        elif self._state == DeviceState.FREE:
            earliest_ready_op = self._get_earlist_ready_op(ready_op_manager)
            if earliest_ready_op is not None:
                next_ts = earliest_ready_op['schedule_tss'][self.id]
        else:
            raise ValueError('Unknown state: {}'.format(self._state))
        if next_ts is not None and next_ts <= self._current_ts:
            raise ValueError('Timestamp should move forward')
        return next_ts

    def run_op(self, op_id):
        """Runs the given op on this device.
        Update op information and device information for this execution.
        """
        op_data = self._op_graph.nodes[op_id]
        super(ETFDevice, self).run_op(op_id)
        self._last_op = self._op_graph.nodes[op_id]

    ## not used in training mode. useful in inference mode (yet to implement)
    def deallocate_predecessor_memory(self, op_id, devices):
        """Deallocates input op's output memory on finishing its successor ops.
        In inference mode, This should be called after run_op().
        """
        for input_op_id, _ in self._op_graph.in_edges(op_id):
            input_op = self._op_graph.nodes[input_op_id]
            executed_out_count = input_op.get("executed_out_count", 0) + 1
            input_op["executed_out_count"] = executed_out_count
            if executed_out_count == self._op_graph.out_degree(input_op_id):
                # deallocate input op's output memory
                devices[input_op["p"]]._deallocate_memory_raw(
                    sum(input_op["output_mem"]), input_op)


ScheduleOpMetadata = namedtuple(
    'ScheduleOpMetadata', ['ts_op_tuple', 'device'])


class ETF():
    """Memory-constrainted earliest time first placement."""

    def __init__(self, op_graph, device_graph,
                 log_file=None):
        # pylint: disable=too-many-arguments
        self.op_graph = op_graph
        self._device_graph = device_graph
        self._log_file = log_file

        # initialized in self.initialize()
        self._num_scheduled_ops = None
        self._devices = {}  # device id -> device wrapper
        self._ready_op_manager = None
        self._topo_order = 0

    def initialize(self):
        """Initializes."""
        self._num_scheduled_ops = 0
        self._devices = {
            device_id: ETFDevice(device_id, self._device_graph, self.op_graph)
            for device_id in self._device_graph.nodes }
        self._ready_op_manager = placer_utils.ReadyOpManager(
            self.op_graph, self._devices, self._log_file)

        for op_id, op_data in self.op_graph.nodes.items():
            op_data['ready_count'] = 0
            if self.op_graph.in_degree(op_id) == 0:
                self._ready_op_manager.add(op_data)

        #self._topo_order = 0


    def _process_scheduled_op(self, scheduled_op):
        self._ready_op_manager.remove(scheduled_op)
        self._num_scheduled_ops += 1


    def _get_next_op_metadata(self):
        '''
            Get the op top of the op list i.e the op that can be 
            scheduled the earliest on any of the devices.
        '''
        self._ready_op_manager.populate_schedule_tss()

        next_op_metadata = None
        for device in self._devices.values():
            # Get the op that can be scheduled the earliest on this device
            next_op = device.get_op_to_schedule(self._ready_op_manager)
            if next_op is None:
                continue
            assert device.is_placeable(next_op)

            # (op_ts, op, device)
            op_metadata = ScheduleOpMetadata(
                ts_op_tuple=placer_utils.TimestampOpTuple(
                    device.get_schedule_ts(next_op), next_op),
                device=device)

            if next_op_metadata is None:
                next_op_metadata = op_metadata
            else:
                if op_metadata.ts_op_tuple < next_op_metadata.ts_op_tuple:
                    next_op_metadata = op_metadata
        # CHECK: if next_op_metadata is none, there is no op left or
        #        no op can be scheduled on any of the devices
        return next_op_metadata

    def _run_schedule_step(self, timestamp):
        """Runs a single schedule step at the given timestamp.

        Returns:
            next schedule timestamp.
        """
        # For each device extend the time
        # and get the new ops ready to be scheduled
        _LOGGER.info('Timestamp {} started'.format(timestamp))

        for device in self._devices.values():
            self._ready_op_manager.extend(device.advance(timestamp)) 
            _LOGGER.debug('-- In device {}, ready ops = {} '.format(device, self._ready_op_manager._ready_ops))

        while True:
            # Get the op that can be scheduled the earliest on any oof the devices
            op_metadata = self._get_next_op_metadata()
            
            if op_metadata is None:
                # no device has an operator to run at this time.
                break

            op_data = op_metadata.ts_op_tuple.op
            # Get ts when the earliest job is ready (all parent nodes finish
            # + inputs have reached the device)
            op_data['ready_ts'] = self._ready_op_manager.get_ready_ts(
                op_data, op_metadata.device)
            op_id = op_data['id']
            if 'p' not in op_data: # if this job has not been placed yet
                # place it on the device it can be scheduled earliest on
                op_metadata.device.place_op(op_id) 
                _LOGGER.info( 'op {} placed on device {}'.format(op_data['name'], op_metadata.device.id) )
                _LOGGER.info( ('\tMemory stats: \n \tMax mem = {} \n \tUsed mem = {} \n \tAvailable mem = {} \n' 
                              + ' \tPeak mem = {} \n -------------').format(
                                  placer_utils.humanize_num_bytes(op_metadata.device.memory_limit),
                                  placer_utils.humanize_num_bytes(op_metadata.device.used_memory),
                                  placer_utils.humanize_num_bytes(op_metadata.device.available_memory),
                                  placer_utils.humanize_num_bytes(op_metadata.device.peak_memory)
                              ) )
                              
                ### To record the topo_order
                op_data['topo_order'] = self._topo_order
                self._topo_order = self._topo_order + 1

            assert op_data['p'] == op_metadata.device.id
            op_metadata.device.run_op(op_id)

            ####################### 
            if self._log_file:
                self._log_file.write(placer_utils.generate_op_run_log(op_data))
            self._process_scheduled_op(op_data)

        min_next_ts = None
        for device in self._devices.values():
            next_ts = device.get_next_ts(self._ready_op_manager)
            if next_ts is not None:
                min_next_ts = min(min_next_ts or next_ts, next_ts)

        return min_next_ts

    def run(self):
        """Runs the placement."""
        # start to schedule and place ops
        self.initialize()

        current_ts = 0  # current timestamp
        while True:
            next_ts = self._run_schedule_step(current_ts)
            if next_ts is None:
                break
            current_ts = next_ts

        assert self._num_scheduled_ops == self.op_graph.number_of_nodes(), \
            "# scheduled ops={}, # ops={}".format(
                self._num_scheduled_ops, self.op_graph.number_of_nodes())

        return current_ts


def m_etf(op_graph, device_graph):
    """Places operators over the devices by using ETF."""
    etf = ETF(copy.deepcopy(op_graph), copy.deepcopy(device_graph))
    runtime = etf.run()
    _LOGGER.info('ETF estimated runtime: %f', runtime / 1e6)
    placer_utils.transfer_placement(etf.op_graph, op_graph)
    return op_graph