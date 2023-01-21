# pylint: disable=missing-function-docstring,missing-class-docstring
# pylint: disable=protected-access

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import unittest
import networkx as nx

from placer import placer_lib
from placer.deprecated.m_sct_v1 import SCT, DeviceState
from placer.placer_utils import find_index_of_ts_op_tuple, assign_topo_order


class SCTTest(unittest.TestCase):

    def __init__(self, *args, **kwargs):
        super(SCTTest, self).__init__(*args, **kwargs)

    def test_assign_favorite_child(self):
        devices = {0: {'name': '/GPU:0', 'memory_size': 10}}
        device_graph = placer_lib.create_device_graph(devices)
        # [case 1]
        # op1 -0(*)-> op3     * : favorite child edge
        #       /
        #      1
        # op2 /
        favorite_child_edges = [0, 1]
        op_graph = nx.DiGraph()
        op_graph.add_edge(1, 3, id=0)
        op_graph.add_edge(2, 3, id=1)

        sct = SCT(op_graph, device_graph)
        sct.assign_favorite_child(favorite_child_edges)

        self.assertEqual(sct.op_graph.nodes[1]['favorite'], 3)
        self.assertEqual(sct.op_graph.nodes[2]['favorite'], -1)
        self.assertEqual(sct.op_graph.nodes[3]['favorite'], -1)
        self.assertEqual(sct.op_graph.nodes[1]['parent'], -1)
        self.assertEqual(sct.op_graph.nodes[2]['parent'], -1)
        self.assertEqual(sct.op_graph.nodes[3]['parent'], 1)

        # [case 2]
        # op1 -0-> op3
        #       /
        #      1
        # op2 /
        favorite_child_edges = [1, 1]
        op_graph = nx.DiGraph()
        op_graph.add_edge(1, 3, id=0)
        op_graph.add_edge(2, 3, id=1)

        sct = SCT(op_graph, device_graph)
        sct.assign_favorite_child(favorite_child_edges)

        self.assertEqual(sct.op_graph.nodes[1]['favorite'], -1)
        self.assertEqual(sct.op_graph.nodes[2]['favorite'], -1)
        self.assertEqual(sct.op_graph.nodes[3]['favorite'], -1)
        self.assertEqual(sct.op_graph.nodes[1]['parent'], -1)
        self.assertEqual(sct.op_graph.nodes[2]['parent'], -1)
        self.assertEqual(sct.op_graph.nodes[3]['parent'], -1)

        # [case 3]
        # op1 -0(*)-> op3
        #       /
        #      1(*)
        # op2 /
        favorite_child_edges = [0, 0]
        op_graph = nx.DiGraph()
        op_graph.add_edge(1, 3, id=0)
        op_graph.add_edge(2, 3, id=1)

        sct = SCT(op_graph, device_graph)
        sct.assign_favorite_child(favorite_child_edges)

        self.assertEqual(sct.op_graph.nodes[1]['parent'], -1)
        self.assertEqual(sct.op_graph.nodes[2]['parent'], -1)
        op3_parent = sct.op_graph.nodes[3]['parent']
        self.assertIn(op3_parent, [1, 2])
        if op3_parent == 1:
            self.assertEqual(sct.op_graph.nodes[1]['favorite'], 3)
            self.assertEqual(sct.op_graph.nodes[2]['favorite'], -1)
        else:
            self.assertEqual(sct.op_graph.nodes[1]['favorite'], -1)
            self.assertEqual(sct.op_graph.nodes[2]['favorite'], 3)
        self.assertEqual(sct.op_graph.nodes[3]['favorite'], -1)

    @staticmethod
    def create_test_sct_instance():
        devices = {
            0: {'name': '/GPU:0', 'memory_size': 10},
            1: {'name': '/GPU:1', 'memory_size': 10},
        }
        device_graph = placer_lib.create_device_graph(devices)
        # op0(w=2,m=5) --e0(w=1)--> op2(w=5,m=2)
        #                        /
        # op1(w=3,m=6) --e1(w=3)/
        op0 = {'id': 0, 'name': 'op0', 'weight': 2,
               'temporary_memory': 0, 'persistent_memory': 5}
        op1 = {'id': 1, 'name': 'op1', 'weight': 3,
               'temporary_memory': 0, 'persistent_memory': 6}
        op2 = {'id': 2, 'name': 'op2', 'weight': 5,
               'temporary_memory': 0, 'persistent_memory': 2}
        op_graph = nx.DiGraph()
        op_graph.add_node(0, **op0)
        op_graph.add_node(1, **op1)
        op_graph.add_node(2, **op2)
        op_graph.add_edge(0, 2, id=0, weight=1)
        op_graph.add_edge(1, 2, id=1, weight=3)
        assign_topo_order(op_graph)

        sct = SCT(op_graph, device_graph)
        sct.assign_favorite_child([1, 1])
        return sct

    def test_initialize_ops(self):
        sct = self.create_test_sct_instance()
        sct._initialize_ops()

        self.assertEqual(sct.op_graph.nodes[0]['ready_count'], 0)
        self.assertEqual(sct.op_graph.nodes[1]['ready_count'], 0)
        self.assertEqual(sct.op_graph.nodes[2]['ready_count'], 0)
        self.assertEqual(len(sct._urgent_ts_ops), 2)

        urgent_ts_op0 = sct._urgent_ts_ops[
            find_index_of_ts_op_tuple(sct._urgent_ts_ops, 0)]
        self.assertEqual(urgent_ts_op0.ts, 0)
        self.assertEqual(urgent_ts_op0.op_id, 0)
        urgent_ts_op1 = sct._urgent_ts_ops[
            find_index_of_ts_op_tuple(sct._urgent_ts_ops, 1)]
        self.assertEqual(urgent_ts_op1.ts, 0)
        self.assertEqual(urgent_ts_op1.op_id, 1)

        # check devices
        for device in sct.devices:
            self.assertEqual(len(device._ready_ts_ops), 2)
            ready_ts_op0 = device._ready_ts_ops[
                find_index_of_ts_op_tuple(device._ready_ts_ops, 0)]
            self.assertEqual(ready_ts_op0.ts, 0)
            self.assertEqual(ready_ts_op0.op_id, 0)

            ready_ts_op1 = device._ready_ts_ops[
                find_index_of_ts_op_tuple(device._ready_ts_ops, 1)]
            self.assertEqual(ready_ts_op1.ts, 0)
            self.assertEqual(ready_ts_op1.op_id, 1)

    def test_run_schedule_step_wo_favorite_child(self):
        sct = self.create_test_sct_instance()
        sct._initialize_ops()
        # op0(w=2,m=5,p=0) --e0(w=1)--> op2(w=5,m=2,p=1)
        #                            /
        # op1(w=3,m=6,p=1) --e1(w=3)/
        device0 = sct.devices[0]
        device1 = sct.devices[1]
        op0 = sct.op_graph.nodes[0]
        op1 = sct.op_graph.nodes[1]
        op2 = sct.op_graph.nodes[2]

        next_ts = sct._run_schedule_step(0)  # ts = 0
        self.assertEqual(next_ts, 2)
        self.assertEqual(op0['p'], 0)
        self.assertEqual(op0['start_ts'], 0)
        self.assertEqual(device0.next_available_ts, 2)
        self.assertEqual(device0.used_memory, 5)
        self.assertEqual(op1['p'], 1)
        self.assertEqual(op1['start_ts'], 0)
        self.assertEqual(device1.next_available_ts, 3)
        self.assertEqual(device1.used_memory, 6)

        next_ts = sct._run_schedule_step(next_ts)  # ts = 2
        self.assertEqual(next_ts, 3)
        self.assertEqual(device0._state, DeviceState.FREE)

        next_ts = sct._run_schedule_step(next_ts)  # ts = 3
        self.assertEqual(next_ts, 8)
        self.assertEqual(op2['p'], 1)
        self.assertEqual(op2['start_ts'], 3)
        self.assertEqual(device1.next_available_ts, 8)
        self.assertEqual(device1.used_memory, 8)

        next_ts = sct._run_schedule_step(next_ts)  # ts = 8
        self.assertIsNone(next_ts)

    def test_run_schedule_wo_favorite_child(self):
        sct = self.create_test_sct_instance()
        # op0(w=2,m=5,p=0) --e0(w=1)--> op2(w=5,m=2,p=1)
        #                            /
        # op1(w=3,m=6,p=1) --e1(w=3)/
        device0 = sct.devices[0]
        device1 = sct.devices[1]
        op0 = sct.op_graph.nodes[0]
        op1 = sct.op_graph.nodes[1]
        op2 = sct.op_graph.nodes[2]

        ts = sct._run_schedule()
        self.assertEqual(ts, 8)
        self.assertEqual(op0['p'], 0)
        self.assertEqual(op0['start_ts'], 0)
        self.assertEqual(op0['end_ts'], 2)
        self.assertEqual(op1['p'], 1)
        self.assertEqual(op1['start_ts'], 0)
        self.assertEqual(op1['end_ts'], 3)
        self.assertEqual(op2['p'], 1)
        self.assertEqual(op2['start_ts'], 3)
        self.assertEqual(op2['end_ts'], 8)
        self.assertEqual(device0.used_memory, 5)
        self.assertEqual(device1.used_memory, 8)

    def test_run_schedule_step_w_favorite_child(self):
        devices = {
            0: {'name': '/GPU:0', 'memory_size': 10},
            1: {'name': '/GPU:1', 'memory_size': 10},
        }
        device_graph = placer_lib.create_device_graph(devices)
        # op0(w=2,m=5) --e0(w=3)--> op1(w=5,m=3)
        #              \
        #               \e1(w=1)--> op2(w=3.m=2)
        op0 = {'id': 0, 'name': 'op0', 'weight': 2,
               'temporary_memory': 0, 'persistent_memory': 5}
        op1 = {'id': 1, 'name': 'op1', 'weight': 5,
               'temporary_memory': 0, 'persistent_memory': 3}
        op2 = {'id': 2, 'name': 'op2', 'weight': 3,
               'temporary_memory': 0, 'persistent_memory': 2}
        op_graph = nx.DiGraph()
        op_graph.add_node(0, **op0)
        op_graph.add_node(1, **op1)
        op_graph.add_node(2, **op2)
        op_graph.add_edge(0, 1, id=0, weight=3)
        op_graph.add_edge(0, 2, id=1, weight=1)
        assign_topo_order(op_graph)

        sct = SCT(op_graph, device_graph)
        sct.assign_favorite_child([0, 1])
        sct._initialize_ops()
        # op0(w=2,m=5,p=0) --e0(w=3,*)--> op1(w=5,m=3,p=0)
        #                   \
        #                    \e1(w=1)--> op2(w=3.m=2,p=1)
        device0 = sct.devices[0]
        device1 = sct.devices[1]
        op0 = sct.op_graph.nodes[0]
        op1 = sct.op_graph.nodes[1]
        op2 = sct.op_graph.nodes[2]

        next_ts = sct._run_schedule_step(0)  # ts = 0
        self.assertEqual(next_ts, 2)
        self.assertEqual(device0._state, DeviceState.BUSY)
        self.assertEqual(device1._state, DeviceState.FREE)
        self.assertEqual(op0['p'], 0)
        self.assertEqual(op0['start_ts'], 0)
        self.assertEqual(device0.next_available_ts, 2)
        self.assertEqual(device0.used_memory, 5)

        next_ts = sct._run_schedule_step(next_ts)  # ts = 2
        self.assertEqual(next_ts, 3)
        self.assertEqual(device0._state, DeviceState.BUSY)
        self.assertEqual(device1._state, DeviceState.FREE)
        self.assertEqual(op1['p'], 0)
        self.assertEqual(op1['start_ts'], 2)
        self.assertEqual(device0.next_available_ts, 7)
        self.assertEqual(device0.used_memory, 8)

        next_ts = sct._run_schedule_step(next_ts)  # ts = 3
        self.assertEqual(next_ts, 6)
        self.assertEqual(device0._state, DeviceState.BUSY)
        self.assertEqual(device1._state, DeviceState.BUSY)
        self.assertEqual(op2['p'], 1)
        self.assertEqual(op2['start_ts'], 3)
        self.assertEqual(device1.next_available_ts, 6)
        self.assertEqual(device1.used_memory, 2)

        next_ts = sct._run_schedule_step(next_ts)  # ts = 6
        self.assertEqual(next_ts, 7)
        self.assertEqual(device0._state, DeviceState.BUSY)
        self.assertEqual(device1._state, DeviceState.FREE)

        next_ts = sct._run_schedule_step(next_ts)  # ts = 7
        self.assertIsNone(next_ts)
        self.assertEqual(device0._state, DeviceState.FREE)
        self.assertEqual(device1._state, DeviceState.FREE)

    def test_run_schedule_step_w_favorite_child2(self):
        devices = {
            0: {'name': '/GPU:0', 'memory_size': 10},
            1: {'name': '/GPU:1', 'memory_size': 10},
        }
        device_graph = placer_lib.create_device_graph(devices)
        op0 = {'id': 0, 'name': 'op0', 'weight': 2,
               'temporary_memory': 0, 'persistent_memory': 5}
        op1 = {'id': 1, 'name': 'op1', 'weight': 3,
               'temporary_memory': 0, 'persistent_memory': 3}
        op2 = {'id': 2, 'name': 'op2', 'weight': 3,
               'temporary_memory': 0, 'persistent_memory': 2}
        op_graph = nx.DiGraph()
        op_graph.add_node(0, **op0)
        op_graph.add_node(1, **op1)
        op_graph.add_node(2, **op2)
        op_graph.add_edge(0, 2, id=0, weight=2)
        op_graph.add_edge(1, 2, id=1, weight=1)
        assign_topo_order(op_graph)
        sct = SCT(op_graph, device_graph)
        sct.assign_favorite_child([1, 0])
        sct._initialize_ops()
        # TODO: find better example
        # op0(w=2,m=5,p=0) --e0(w=2)--
        #                             \
        # op1(w=3,m=3,p=1) --e1(w=1,*)--> op2(w=3,m=2,p=1)
        device0 = sct.devices[0]
        device1 = sct.devices[1]
        op0 = sct.op_graph.nodes[0]
        op1 = sct.op_graph.nodes[1]
        op2 = sct.op_graph.nodes[2]

        next_ts = sct._run_schedule_step(0)  # ts = 0
        self.assertEqual(next_ts, 2)
        self.assertEqual(device0._state, DeviceState.BUSY)
        self.assertEqual(device1._state, DeviceState.BUSY)
        self.assertEqual(op0['p'], 0)
        self.assertEqual(op0['start_ts'], 0)
        self.assertEqual(device0.next_available_ts, 2)
        self.assertEqual(device0.used_memory, 5)
        self.assertEqual(op1['p'], 1)
        self.assertEqual(op1['start_ts'], 0)
        self.assertEqual(device1.next_available_ts, 3)
        self.assertEqual(device1.used_memory, 3)

        next_ts = sct._run_schedule_step(next_ts)  # ts = 2
        self.assertEqual(next_ts, 3)
        self.assertEqual(device0._state, DeviceState.FREE)
        self.assertEqual(device1._state, DeviceState.BUSY)

        next_ts = sct._run_schedule_step(next_ts)  # ts = 3
        self.assertEqual(next_ts, 4)
        self.assertEqual(device0._state, DeviceState.FREE)
        self.assertEqual(device1._state, DeviceState.AWAKE)

        next_ts = sct._run_schedule_step(next_ts)  # ts = 4
        self.assertEqual(next_ts, 7)
        self.assertEqual(device0._state, DeviceState.BUSY)
        self.assertEqual(device1._state, DeviceState.FREE)
        self.assertEqual(op2['p'], 0)
        self.assertEqual(op2['start_ts'], 4)
        self.assertEqual(device0.next_available_ts, 7)
        self.assertEqual(device0.used_memory, 7)

        next_ts = sct._run_schedule_step(next_ts)  # ts = 7
        self.assertIsNone(next_ts)
        self.assertEqual(device0._state, DeviceState.FREE)
        self.assertEqual(device1._state, DeviceState.FREE)

    def test_run_schedule_step_w_favorite_child3(self):
        devices = {
            0: {'name': '/GPU:0', 'memory_size': 10},
            1: {'name': '/GPU:1', 'memory_size': 10},
        }
        device_graph = placer_lib.create_device_graph(devices)
        op0 = {'id': 0, 'name': 'op0', 'weight': 2,
               'temporary_memory': 0, 'persistent_memory': 5}
        op1 = {'id': 1, 'name': 'op1', 'weight': 3,
               'temporary_memory': 0, 'persistent_memory': 3}
        op2 = {'id': 2, 'name': 'op2', 'weight': 3,
               'temporary_memory': 0, 'persistent_memory': 2}
        op_graph = nx.DiGraph()
        op_graph.add_node(0, **op0)
        op_graph.add_node(1, **op1)
        op_graph.add_node(2, **op2)
        op_graph.add_edge(0, 2, id=0, weight=2)
        op_graph.add_edge(1, 2, id=1, weight=1)
        assign_topo_order(op_graph)
        sct = SCT(op_graph, device_graph)
        sct.assign_favorite_child([0, 1])
        sct._initialize_ops()
        # op0(w=2,m=5,p=0) --e0(w=2,*)
        #                             \
        # op1(w=3,m=3,p=1) --e1(w=1)--> op2(w=3,m=2,p=0)
        device0 = sct.devices[0]
        device1 = sct.devices[1]
        op0 = sct.op_graph.nodes[0]
        op1 = sct.op_graph.nodes[1]
        op2 = sct.op_graph.nodes[2]

        next_ts = sct._run_schedule_step(0)  # ts = 0
        self.assertEqual(next_ts, 2)
        self.assertEqual(device0._state, DeviceState.BUSY)
        self.assertEqual(device1._state, DeviceState.BUSY)
        self.assertEqual(op0['p'], 0)
        self.assertEqual(op0['start_ts'], 0)
        self.assertEqual(device0.next_available_ts, 2)
        self.assertEqual(device0.used_memory, 5)
        self.assertEqual(op1['p'], 1)
        self.assertEqual(op1['start_ts'], 0)
        self.assertEqual(device1.next_available_ts, 3)
        self.assertEqual(device1.used_memory, 3)

        next_ts = sct._run_schedule_step(next_ts)  # ts = 2
        self.assertEqual(next_ts, 3)
        self.assertEqual(device0._state, DeviceState.FREE)
        self.assertEqual(device1._state, DeviceState.BUSY)

        next_ts = sct._run_schedule_step(next_ts)  # ts = 3
        self.assertEqual(next_ts, 4)
        self.assertEqual(device0._state, DeviceState.AWAKE)
        self.assertEqual(device1._state, DeviceState.FREE)

        next_ts = sct._run_schedule_step(next_ts)  # ts = 4
        self.assertEqual(next_ts, 7)
        self.assertEqual(device0._state, DeviceState.BUSY)
        self.assertEqual(device1._state, DeviceState.FREE)
        self.assertEqual(op2['p'], 0)
        self.assertEqual(op2['start_ts'], 4)
        self.assertEqual(device0.next_available_ts, 7)
        self.assertEqual(device0.used_memory, 7)

        next_ts = sct._run_schedule_step(next_ts)  # ts = 7
        self.assertIsNone(next_ts)
        self.assertEqual(device0._state, DeviceState.FREE)
        self.assertEqual(device1._state, DeviceState.FREE)


if __name__ == '__main__':
    unittest.main()
