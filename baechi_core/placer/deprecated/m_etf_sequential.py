"""Memory constrainted earliest time first placement algorithm."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import copy

from placer import device as device_wrapper
from placer import placer_utils as utils
from utils import logger

_LOGGER = logger.get_logger(__file__, level=logger.INFO)


class ETF():
    """Memory constrainted earliest time first placement algorithm."""

    def __init__(self, op_graph, device_graph, log_file=None):
        self.op_graph = op_graph
        self.device_graph = device_graph
        self._log_file = log_file

        self._devices = {
            device_id: device_wrapper.DeviceWrapperAllocator(
                device_id, self.device_graph, self.op_graph, True)
            for device_id in self.device_graph.nodes}
        self._ready_op_manager = utils.ReadyOpManager(
            self.op_graph, self._devices, self._log_file)

    def initialize(self):
        """Initializes."""
        for op_id, op_data in self.op_graph.nodes().items():
            op_data["ready_count"] = 0
            if self.op_graph.in_degree(op_id) == 0:
                self._ready_op_manager.add(op_data)

    def get_next_op(self):
        """Returns a next op triple (op, ready_ts, device)."""
        # TODO: optimize this. avoid ready ts calculation for every call.
        self._ready_op_manager.populate_schedule_tss()

        schedule_ts_op_tuples = utils.SortedTimestampOps()
        for ready_op in self._ready_op_manager:
            schedule_tss = ready_op['schedule_tss']
            min_schedule_ts = min(schedule_tss.values())
            schedule_ts_op_tuples.add_op(min_schedule_ts, ready_op)

        target_ts_op_tuple = schedule_ts_op_tuples.pop()
        target_op = target_ts_op_tuple.op
        schedule_ts = target_ts_op_tuple.ts

        min_schedule_ts, device_id = min(
            [(timestamp, device_id) for device_id, timestamp
             in self._ready_op_manager.get_schedule_tss(target_op).items()])

        assert min_schedule_ts == schedule_ts

        return target_op, self._devices[device_id]

    def run(self):
        """Runs the m-ETF placement."""
        num_scheduled_ops = 0
        while len(self._ready_op_manager) > 0:
            target_op, device = self.get_next_op()
            target_op_id = target_op['id']
            target_op['ready_ts'] = self._ready_op_manager.get_ready_ts(
                target_op, device, dry_run=False)
            device.place_op(target_op_id)
            device.run_op(target_op_id)
            self._ready_op_manager.remove(target_op)
            num_scheduled_ops += 1

            if self._log_file:
                self._log_file.write(utils.generate_op_run_log(target_op))

            # check whether output ops are ready to run
            self._ready_op_manager.extend(
                utils.process_finished_op(self.op_graph, target_op_id))

        assert num_scheduled_ops == self.op_graph.number_of_nodes(), \
            "# scheduled ops={}, # ops={}".format(
                num_scheduled_ops, self.op_graph.number_of_nodes())

        runtime = max([device.next_available_ts
                       for device in self._devices.values()])
        _LOGGER.info('ETF estimated runtime: %f', runtime / 1e6)


class ETFWithColocation(object):
    """Place ops into devices by ETF algorithm with colocation rules."""

    def __init__(self, op_graph, device_graph):
        self.op_graph = op_graph
        self.device_graph = device_graph
        self.devices = {}
        self.ready_op_list = []
        self.colocation_group_infos = {}

    def initialize(self):
        self.devices = {device_id: utils.DeviceWrapper(device_data)
                        for device_id, device_data
                        in self.device_graph.nodes.items()}

        self.colocation_group_infos = utils.create_colocation_group_infos(
            self.op_graph)

        for op_id, op_data in self.op_graph.nodes().items():
            op_data["ready_count"] = 0
            if self.op_graph.in_degree(op_id) == 0:
                # ready_times is a ReadyTimeInfo list
                op_data["ready_times"] = self.get_ready_times(op_data)
                self.ready_op_list.append(op_data)

        _LOGGER.debug("Ready ops: %s",
                      str([op["name"] for op in self.ready_op_list]))

    def get_ready_times(self, op):
        """Returns a list of ready times for all available devices."""
        ready_times = []
        for device in self.devices.values():
            ready_time = self.get_ready_time(op, device)
            if ready_time is not None:
                ready_times.append(ReadyTimeInfo(ready_time, device.id))
        return ready_times

    def get_ready_time(self, op, device):
        """Returns a ready time for the given device."""
        if "p" not in op:
            # check whether all ops in the colocation group of the op can be
            # placed.
            group_info = self.colocation_group_infos[op["colocation_group"]]
            if group_info["memory"] > device.available_memory:
                # cannot place this op on the given device
                return None
        else:
            # this op was already placed on some device
            # do no need to check memory (see DeviceWrapper.place_op())
            if op["p"] != device.id:
                return None

        ready_time = device.next_available_ts
        for in_edge in self.op_graph.in_edges(op["id"], data=True):
            from_op_id, _, edge_data = in_edge
            from_op = self.op_graph.nodes[from_op_id]
            data_transfer_time = (
                0 if from_op["p"] == device.id else edge_data["weight"])
            ready_time = max(ready_time,
                             from_op["end_time"] + data_transfer_time)

        return ready_time

    def run_op(self, op, ready_time_info):
        device = self.devices[ready_time_info.device_id]
        ready_time = ready_time_info.time

        assert device.next_available_ts <= ready_time, \
            "ready time op={}, device={}".format(
                ready_time, device.next_available_ts)

        # place op first if necessary
        if "p" not in op:
            # this op was not placed before.
            # place all ops in the same colocation group
            group_info = self.colocation_group_infos[op["colocation_group"]]
            for colocated_op in group_info["ops"]:
                device.place_op(colocated_op)

                if "ready_times" in colocated_op:
                    # remove ready times for other devices.
                    # assign a dummy ready time for this device.
                    # this dummy will be updated below.
                    colocated_op["ready_times"] = [
                        ReadyTimeInfo(float('inf'), device.id)]

        device.run_op(op, ready_time)

        # update existing ready times
        for ready_op in self.ready_op_list:
            ready_times = ready_op['ready_times']
            i = find_ready_time_by_device_id(ready_times, device.id)
            if i == -1:
                continue

            new_ready_time = self.get_ready_time(ready_op, device)
            if new_ready_time is None:
                # ready_op cannot run on this device
                del ready_times[i]
            else:
                # update ready time for this device
                ready_times[i] = ReadyTimeInfo(new_ready_time, device.id)

        # check whether its next ops are ready
        for _, next_op_id in self.op_graph.out_edges(op["id"]):
            next_op = self.op_graph.nodes[next_op_id]
            next_op["ready_count"] += 1
            if next_op["ready_count"] == self.op_graph.in_degree(next_op_id):
                # ready to run
                self.ready_op_list.append(next_op)
                next_op["ready_times"] = self.get_ready_times(next_op)
                _LOGGER.debug("Op name=%s is ready. ready_times=%s.",
                              next_op["name"], next_op["ready_times"])

        return True

    def get_next_op(self):
        """Returns a op that can schedule at the earlist time."""

        earlist_ready_op = None
        earlist_ready_op_time = ReadyTimeInfo(float("inf"), 0)

        for ready_op in self.ready_op_list:
            ready_op_time = min(ready_op["ready_times"])
            if ready_op_time < earlist_ready_op_time:
                earlist_ready_op = ready_op
                earlist_ready_op_time = ready_op_time

        # multiple devices may have the same ready time for the given op
        earlist_ready_op_times = [
            ready_time
            for ready_time in earlist_ready_op["ready_times"]
            if ready_time.time == earlist_ready_op_time.time]

        self.ready_op_list.remove(earlist_ready_op)

        return earlist_ready_op, earlist_ready_op_times

    def run(self):
        while self.ready_op_list:
            target_op, ready_times = self.get_next_op()

            _LOGGER.debug(
                "Next op to schedule name=%s, ready_times=%s",
                target_op["name"],
                str(ready_times))

            # pick the first ready device
            self.run_op(target_op, ready_times[0])


def m_etf(op_graph, device_graph, colocation=False):
    """Places operators on devices according to the ETF algorithm."""
    etf_cls = ETFWithColocation if colocation else ETF

    etf = etf_cls(copy.deepcopy(op_graph), copy.deepcopy(device_graph))
    etf.initialize()
    etf.run()

    utils.transfer_placement(etf.op_graph, op_graph)

    return op_graph
