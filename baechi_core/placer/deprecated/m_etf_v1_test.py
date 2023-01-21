# pylint: disable=missing-function-docstring,missing-class-docstring
# pylint: disable=protected-access

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import unittest
import networkx as nx

from placer import placer_lib
from placer.placer_utils import find_index_of_ts_op_tuple, assign_topo_order
from placer.deprecated.m_etf_v1 import DeviceWrapper, DeviceState
from placer.deprecated.m_etf_v1 import ETF


class DeviceWrapperTest(unittest.TestCase):

    def test_add_ready_op(self):
        op1 = {'id': 1, 'name': 'op1', 'memory': 5, 'ready_tss': {0: 0}}
        op2 = {'id': 2, 'name': 'op2', 'memory': 2, 'ready_tss': {0: 0}}
        op3 = {'id': 3, 'name': 'op2', 'memory': 12, 'ready_tss': {0: 0}}
        op_graph = nx.DiGraph()
        op_graph.add_node(1, **op1)
        op_graph.add_node(2, **op2)
        op_graph.add_node(3, **op3)

        device_graph = placer_lib.create_device_graph(
            {0: {'name': '/GPU:0', 'memory_size': 10}})
        device = DeviceWrapper(0, device_graph, op_graph)

        self.assertTrue(device.add_ready_op(op1))
        self.assertEqual(len(device._ready_ts_ops), 1)

        self.assertTrue(device.add_ready_op(op2))
        self.assertEqual(len(device._ready_ts_ops), 2)

        self.assertFalse(device.add_ready_op(op3))
        self.assertEqual(len(device._ready_ts_ops), 2)


class ETFTest(unittest.TestCase):

    @staticmethod
    def create_test_etf_instance():
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

        return ETF(op_graph, device_graph)

    def test_initialize_ops(self):
        etf = self.create_test_etf_instance()
        etf._initialize_ops()

        self.assertEqual(etf.op_graph.nodes[0]['ready_count'], 0)
        self.assertEqual(etf.op_graph.nodes[1]['ready_count'], 0)
        self.assertEqual(etf.op_graph.nodes[2]['ready_count'], 0)

        # check devices
        for device in etf.devices:
            self.assertEqual(len(device._ready_ts_ops), 2)
            ready_ts_op0 = device._ready_ts_ops[
                find_index_of_ts_op_tuple(device._ready_ts_ops, 0)]
            self.assertEqual(ready_ts_op0.ts, 0)
            self.assertEqual(ready_ts_op0.op_id, 0)

            ready_ts_op1 = device._ready_ts_ops[
                find_index_of_ts_op_tuple(device._ready_ts_ops, 1)]
            self.assertEqual(ready_ts_op1.ts, 0)
            self.assertEqual(ready_ts_op1.op_id, 1)

    def test_run_schedule_step(self):
        etf = self.create_test_etf_instance()
        etf._initialize_ops()
        # op0(w=2,m=5,p=0) --e0(w=1)--> op2(w=5,m=2,p=1)
        #                            /
        # op1(w=3,m=6,p=1) --e1(w=3)/
        device0 = etf.devices[0]
        device1 = etf.devices[1]
        op0 = etf.op_graph.nodes[0]
        op1 = etf.op_graph.nodes[1]
        op2 = etf.op_graph.nodes[2]

        next_ts = etf._run_schedule_step(0)  # ts = 0
        self.assertEqual(next_ts, 2)
        self.assertEqual(op0['p'], 0)
        self.assertEqual(op0['start_ts'], 0)
        self.assertEqual(device0.next_available_ts, 2)
        self.assertEqual(device0.used_memory, 5)
        self.assertEqual(op1['p'], 1)
        self.assertEqual(op1['start_ts'], 0)
        self.assertEqual(device1.next_available_ts, 3)
        self.assertEqual(device1.used_memory, 6)

        next_ts = etf._run_schedule_step(next_ts)  # ts = 2
        self.assertEqual(next_ts, 3)
        self.assertEqual(device0._state, DeviceState.FREE)

        next_ts = etf._run_schedule_step(next_ts)  # ts = 3
        self.assertEqual(next_ts, 8)
        self.assertEqual(op2['p'], 1)
        self.assertEqual(op2['start_ts'], 3)
        self.assertEqual(device1.next_available_ts, 8)
        self.assertEqual(device1.used_memory, 8)

        next_ts = etf._run_schedule_step(next_ts)  # ts = 8
        self.assertIsNone(next_ts)

    def test_run_schedule(self):
        etf = self.create_test_etf_instance()
        # op0(w=2,m=5,p=0) --e0(w=1)--> op2(w=5,m=2,p=1)
        #                            /
        # op1(w=3,m=6,p=1) --e1(w=3)/
        device0 = etf.devices[0]
        device1 = etf.devices[1]
        op0 = etf.op_graph.nodes[0]
        op1 = etf.op_graph.nodes[1]
        op2 = etf.op_graph.nodes[2]

        ts = etf._run_schedule()
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


if __name__ == "__main__":
    unittest.main()
