"""Device Wrapper."""
from __future__ import absolute_import, division, print_function

from utils import logger
from placer import placer_utils as putils

_LOGGER = logger.get_logger(__file__, level=logger.INFO)

#################################### DeviceWrapper ################################################
#
#-------------- Device memory ----------------
# ****** Properties ********
# _node["used_mem"]     = used memory so far
# _node["peak_mem"]     = peak memory usage so far
# _node["memory_limit"] = max memory
# available_memory      = memory_limit - used memory
# 
# ****** Methods ***********
# is_placeable(op)      - True if the op can be palced on this device
# place_op(op)          - places the op on the device and updates the "used_mem" and "peak_mem" 
#                           (including for incoming inputs from other devices)
#---------------- Tesnor Communication -------------------
# ****** Properties ********
# _cached_tensors   - dict of tensors already transferred to this device.
#                     Each tensor has 'send_start_ts','send_end_ts','recv_start_ts','recv_end_ts'
# 
# ****** Methods ***********
# get_cached_tensor(tsr_name) - returns tensor tsr_name if in _cached_tensors, else None
# send_tensor(tsr, from_op_end_ts, actually), recv_tensor(tsr_data, send_start_ts, actually) -
#           returns estimated 'send_start_ts',....'recv_end_ts' for tsr, If actually =True, 
#           tensor timings are updated and tsr is added to  _cached_tensors
# recv_inputs(op_data, actually) - Estimates the memory required for inputs transferred from 
#           other devices, if op were placed on this device. If actually = True, the tensor 
#           timings are also updated using send_tensor,recv_tensor. Note: Memory is updated 
#           seperately in place_op()
# 
#-------------- Device scheduling ----------------
# ****** Properties ********
# _next_available_ts    - time at which the device becomes available next
# 
# ****** Methods ***********
# run_op(op_id)  - Once placed in this device, runs the op, updating its start_ts, end_ts and
#                    tensor transfer timings dor its inputs
#
##################################################################################################

class DeviceWrapper():
    """Wrapper class for a node in the device graph."""
    # pylint: disable=too-many-instance-attributes

    def __init__(self, device_id, device_graph, op_graph, memory_check=True):
        self._id = device_id
        self._device_graph = device_graph
        self._node = device_graph.nodes[device_id]
        self._op_graph = op_graph
        self._memory_check = memory_check  # is the device memory_contrained

        self._next_available_ts = 0
        self._cached_tensors = {}

    @staticmethod
    def type():
        """Wrapper type."""
        return "normal"

    @property
    def id(self):
        """Device id."""
        # pylint: disable=invalid-name
        return self._id

    def __getitem__(self, key):
        return self._node[key]

    @property
    def used_memory(self):
        """Currently used memory."""
        return self._node["used_mem"]

    @property
    def peak_memory(self):
        """The peak memory usage so far."""
        return self._node["peak_mem"]

    @property
    def memory_limit(self):
        """Memory limit."""
        return self._node["memory_limit"]

    @property
    def available_memory(self):
        """Available memory on this device."""
        return self.memory_limit - self.used_memory

    @property
    def next_available_ts(self):
        """Device's next available timestamp."""
        return self._next_available_ts


    ##################################### Communication functions #############################################
    def get_cached_tensor(self, tensor_name):
        """Returns whether the given tensor is cached at this device."""
        return self._cached_tensors.get(tensor_name, None)

    def send_tensor(self, tensor_data, from_op_end_ts, actually):
        """Sends the tensor associated with the given edge.
        If actually=True, tensor is considered to be actually transfered and 
        timings are updated, else timing is just computed and returned
        """
        send_start_ts = from_op_end_ts  # Transfer starts on stream as soon as the from_op finishes
        comm_cost = tensor_data['transfer_time']
        send_end_ts = send_start_ts + comm_cost

        if actually:
            if 'send_start_ts' in tensor_data:
                # timings for from_op_dev to any child device is assumed same
                # (i.e parent op sends tensors to all child device as soon as it is done executing)
                assert tensor_data['send_start_ts'] == send_start_ts, 'tenosr timings do not match'
                assert tensor_data['send_end_ts'] == send_end_ts, 'tenosr timings do not match'
            tensor_data['send_start_ts'] = send_start_ts
            tensor_data['send_end_ts'] = send_end_ts
        return send_start_ts, send_end_ts

    def can_recv(self, tensor_data):
        """Checks if the device has enough memory to rcv the tensor"""
        return tensor_data['bytes'] <= self.available_memory

    # TODO: multiple simultaneous reception at a device get queued up.
    def recv_tensor(self, tensor_data, send_start_ts, actually):
        """Receives the given tensor at this device. 
        If actually=True, tensor is considered to be actually transfered and 
        timings are updated, else timing is just computed and returned. 
        Note: this only updates the send/rcv times and local cache on the device. 
        Device memory must be updated seperately in place_op().
        """
        tensor_name = tensor_data['name']
        assert tensor_name not in self._cached_tensors, \
            'tensor {} was already received at device {}'.format(
                tensor_name, self.id)

        if self._memory_check:
            assert self.can_recv(tensor_data),\
                'not enough memory to rcv tensor {} at device {}'.format(tensor_name, self.id)

        comm_cost = tensor_data['transfer_time']
        recv_start_ts = send_start_ts # Reception is assumed to begin as soon as send starts
        if 'send_start_ts' in tensor_data:
            assert recv_start_ts == tensor_data['send_start_ts'], 'tenosr timings do not match'

        recv_end_ts = recv_start_ts + comm_cost

        if actually:
            if 'recv_start_ts' in tensor_data: 
                # timings for from_op_dev to any child device is assumed same
                # (i.e parent op sends tensors to all child device as soon as it is done executing)
                assert tensor_data['recv_start_ts'] == recv_start_ts, 'tenosr timings do not match'
                assert tensor_data['recv_end_ts'] == recv_end_ts, 'tenosr timings do not match'
            else:
                tensor_data['recv_start_ts'] = recv_start_ts
                tensor_data['recv_end_ts'] = recv_end_ts
            self._cached_tensors[tensor_name] = {
                    'send_start_ts': tensor_data['send_start_ts'],
                    'send_end_ts': tensor_data['send_end_ts'],
                    'recv_start_ts': tensor_data['recv_start_ts'],
                    'recv_end_ts': tensor_data['recv_end_ts']}

        return recv_start_ts, recv_end_ts

    ######################################## Memory functions #################################################
    def recv_inputs(self, op_data, actually):
        """If given op is on this device, how much memory must be reserved for inputs copied 
        from other devices. If actually=True, the tensor timings are updated as if the inputs
        are actually moved to current device (used in run_op).
        """
        net_input_mem = 0 #input_mem_estimate
        in_edges = list(self._op_graph.in_edges(op_data['id'], data=True))
        for in_edge in in_edges:
            from_op_id, _, edge_data = in_edge
            from_op = self._op_graph.nodes[from_op_id]
            from_op_device_id = from_op['p']
            if from_op_device_id != self.id:
                if edge_data['tensor']['name'] in self._cached_tensors: #check if input has already been recieved
                    _LOGGER.debug("op {} on dev {}. input tensor {} already on device".format(op_data["id"], 
                    self.id, edge_data['tensor']['name']))
                else: # Bug fixed (vs baechi_demo_v4)
                    net_input_mem += edge_data['tensor']['bytes']
                    _LOGGER.debug("op {} on dev {}. input tensor {} must be rcvd on device. Will occupy {} bytes".format(
                        op_data["id"], self.id, edge_data['tensor']['name'], 
                        putils.humanize_num_bytes(edge_data['tensor']['bytes']) ))
                    if actually:
                        send_start_ts, _ = self.send_tensor(edge_data['tensor'], from_op["end_ts"], actually=True)
                        self.recv_tensor(edge_data['tensor'], send_start_ts, actually=True)
                        _LOGGER.debug("tensor {} transferred from device {} to device {}. mem occupied = {}".format(
                            edge_data['tensor']['name'], from_op_id, self.id, 
                            putils.humanize_num_bytes(edge_data['tensor']['bytes']) ))

        return net_input_mem

    def is_placeable(self, op_data): 
        """Returns whether the given operator can be placed on this device."""
        ## Step 1: Calculate storage required to store inputs to op coming from other devices
        net_input_mem = self.recv_inputs(op_data, actually=False)

        _LOGGER.debug("op {} on dev {}. net input mem required={}".format(
                    op_data["id"], self.id, putils.humanize_num_bytes(net_input_mem) ))
        #####################################################
        ## Step 2: To place a node on a device D, check:
        ## 1. D_used_mem + net_input_mem + node.peak_mem < D_max_mem       # scenario when node is executing
        ## 2. D_peak_mem + net_input_mem + node.parameter_mem < D_max_mem  # scenario when some previous node on D is executing
        ## (where, D_peak_mem is peak memory usage on D until placing this node,
        ## node.parameter_mem is mem for parameters and gradients only. Doens't include output_mem unlike in used_mem
        ## because the output of current layer would have been discarded when a previous layer peaks duding backprop)
        check1 = ( self.used_memory + net_input_mem + op_data["peak_mem"] <= self.memory_limit )
        #check2 = ( self.peak_memory + net_input_mem + op_data["used_mem"] < self.memory_limit )
        check2 = ( self.peak_memory + net_input_mem + op_data["parameter_mem"] <= self.memory_limit )

        _LOGGER.debug("op {} on dev {}. check1={}, check2={}".format(
                    op_data["name"], self.id, check1, check2))

        placeable = (check1 and check2)    
        if not placeable:
            _LOGGER.debug("op {} on dev {}. Is NOT placeable. Device peak mem ={}, op permanent mem={} ".format(
                    op_data["name"], self.id, putils.humanize_num_bytes(self.peak_memory)
                    , putils.humanize_num_bytes(op_data["permanent_mem"]) ))
        return placeable


    def _allocate_memory_raw(self, permanent_num_bytes, temporary_num_bytes=0): 
        """Allocates the given num_bytes and updates the used and peak mem of the device .""" 
        ## Place the node on D and update D_used_mem and D_peak_mem:
        if self._memory_check:
            assert self.available_memory >= permanent_num_bytes + temporary_num_bytes # same as check 1 in is_placeable()

        #update peak_memory
        peak_mem_estimate = self.used_memory + permanent_num_bytes + temporary_num_bytes
        if peak_mem_estimate > self.peak_memory:
            self._node["peak_mem"] = peak_mem_estimate
            _LOGGER.debug("device = {}. updated peak memory. peak_mem={}".format( self.id, 
            putils.humanize_num_bytes(self.peak_memory) ) )

        self._node["used_mem"] += permanent_num_bytes 
        _LOGGER.debug("device = {}. allocated raw memory. num_bytes={}. used_mem={}".format(
                        self.id, putils.humanize_num_bytes(permanent_num_bytes), 
                        putils.humanize_num_bytes(self.used_memory) ) )
                               

    def _deallocate_memory_raw(self, num_bytes):
        if num_bytes == 0:
            return
        self._node["used_mem"] -= num_bytes
        assert self._node["used_mem"]>=0
        _LOGGER.debug('device = {}. deallocated raw memory. num_bytes={}. used_mem={}'.format(self.id,
        putils.humanize_num_bytes(num_bytes), putils.humanize_num_bytes(self.used_memory) ) )
        

    def place_op(self, op_id):
        """Place the given op on this device-> update memory usuage,
         including the cross-device input transfers. Timings are updated in run_op"""
        op_data = self._op_graph.nodes[op_id]

        assert "p" not in op_data, \
            "Operator id={} was already placed. prev={}, new={}".format(
                op_id, op_data["p"], self.id)

        if self._memory_check:
            assert self.is_placeable(op_data), \
                "Cannot place op {} on dev {}".format(op_id, self.id)
        
        op_data["p"] = self.id
        # get mem required for inputs of this op copied from other devices to this device
        net_input_mem = self.recv_inputs(op_data, actually=False)  # timings are actually updated in run_op()
        self._allocate_memory_raw(op_data["permanent_mem"], op_data["temporary_mem"])
        self._allocate_memory_raw(net_input_mem, 0)

    ###################################### Scheduling #############################################
    def run_op(self, op_id):
        """Run the given op on this device.
        Update op and device timing information for this execution.
        """
        op_data = self._op_graph.nodes[op_id]

        assert 'start_ts' not in op_data, \
            "Op id=%d was executed before" % op_id
        assert op_data["p"] == self.id, \
            "Op id={}, Assigned dev={}, this dev={}".format(
                op_id, op_data["p"], self.id)

        # update timing info
        _ = self.recv_inputs(op_data, actually=True) # update timings for input tensors
        op_data["start_ts"] = max(self.next_available_ts, op_data["ready_ts"])
        op_data["end_ts"] = op_data["start_ts"] + op_data["weight"]
        self._next_available_ts = op_data["end_ts"]

        _LOGGER.debug(
            "Op name=%s runs on device %d. start_ts=%.2f, end_ts=%.2f"
            + (", ops=%s" % op_data["op_names"]
               if "op_names" in op_data else ""),
            op_data['name'],
            self.id,
            op_data['start_ts'],
            op_data["end_ts"])

#####################################################################################
