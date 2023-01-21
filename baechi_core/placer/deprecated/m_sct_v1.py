"""Shortest communication time placement."""
# pylint: disable=missing-function-docstring,missing-class-docstring
# pylint: disable=invalid-name
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import copy
import cvxopt

from placer.deprecated import m_etf_v1
from placer import placer_utils
from utils import logger

_LOGGER = logger.get_logger(__file__, level=logger.INFO)

DeviceState = m_etf_v1.DeviceState


class FavoriteChildLPSolver():
    """Solve the linear programming to calculate favorite children.

        LP variables are formatted in [e1,e2,e3,...,n1,n2,n3,...,w].
        e_i represents whether it is a favorite child edge
        n_i represents the start time of each op
        w represents makespan

        Assumes that nodes and edges in the op_graph have ids for each.
        Ids should not have any hole in their sequence.
    """

    def __init__(self, op_graph, threshold=0.5):
        self.op_graph = op_graph
        self._threshold = threshold
        self.constraint_index = 0
        self.LHS_triples = []
        self.RHS = []
        self.num_edges = op_graph.number_of_edges()
        self.num_nodes = op_graph.number_of_nodes()

    @staticmethod
    def get_favorite_child_var_index(edge_id):
        return edge_id

    def get_start_time_var_index(self, node_id):
        return self.num_edges + node_id

    def get_makespan_var_index(self):
        return self.num_edges + self.num_nodes

    def add_constraint(self, var_tuples, value):
        """Add a new constraint.

        Args:
            var_tuples: a list of (var_coeff, var_index)
            value: constant value

        This adds the following constraints.
            {var_coeff_1} * x_{var_index_1} + {var_coeff_2} * x_{var_index_2}
                ... <= {value}
        """
        # a constraint is formatted in LHS <= RHS
        for var_coeff, var_index in var_tuples:
            # (value, row_index, column_index)
            triple = (var_coeff, self.constraint_index, var_index)
            self.LHS_triples.append(triple)
        self.RHS.append(float(value))
        self.constraint_index += 1

    def build_constraints(self):
        # rule 1: the favorite child is either 0 or 1
        #         0 <= e_i (x_i,j) <= 1
        for _, _, edge_id in self.op_graph.edges(data='id'):
            var_index = self.get_favorite_child_var_index(edge_id)
            # e_i <= 1
            self.add_constraint([(1, var_index)], 1)
            # e_i >= 0 (-e_i <= 0)
            self.add_constraint([(-1, var_index)], 0)

        # rule 2: all tasks should start after t=0
        #         s_k >= 0 (-s_k <= 0)
        # s_k has a variable index, {num_edges} + k, in the LP matrix
        for op_id in self.op_graph.nodes():
            self.add_constraint(
                [(-1, self.get_start_time_var_index(op_id))], 0)

        # rule 3: for an edge i -> j, j must start after i completes.
        #         if on different devices, communication cost should be added.
        #         s_i + p_i + c_i,j * x_i,j <= s_j
        for i, j, edge_data in self.op_graph.edges(data=True):
            # s_i - s_j + c_i,j (w(e_i)) * x_i,j (e_i) <= -p_i
            self.add_constraint(
                [(1, self.get_start_time_var_index(i)),
                 (-1, self.get_start_time_var_index(j)),
                 (edge_data['weight'], edge_data['id'])],
                -self.op_graph.nodes[i]['weight'])

        # rule 4: every node has at most one favorite child
        #         for node i, if there is an edge i -> j,
        #         \sum_j x_i,j >= |j| - 1
        for op_id in self.op_graph.nodes():
            # \sum_j - x_i,j <= 1 - |j|
            var_tuples = []
            for _, _, edge_data in self.op_graph.out_edges(op_id, data=True):
                var_tuples.append(
                    (-1, self.get_favorite_child_var_index(edge_data['id'])))
            if var_tuples:
                self.add_constraint(var_tuples, 1 - len(var_tuples))

        # rule 5: every node is the favorite child of at most one predecessor
        #         for node i, if there is an edge j -> i,
        #         \sum_j x_j,i >= |j| - 1
        for op_id in self.op_graph.nodes():
            # \sum_j - x_j,i  <= 1 - |j|
            var_tuples = []
            for _, _, edge_data in self.op_graph.in_edges(op_id, data=True):
                var_tuples.append(
                    (-1, self.get_favorite_child_var_index(edge_data['id'])))
            if var_tuples:
                self.add_constraint(var_tuples, 1 - len(var_tuples))

        # rule 6: all tasks should complete before makespan
        #         s_i + p_i <= w
        for op_id, op_data in self.op_graph.nodes().items():
            # s_i - w <= -p_i
            var_tuples = [(1, self.get_start_time_var_index(op_id)),
                          (-1, self.get_makespan_var_index())]
            self.add_constraint(var_tuples, -op_data['weight'])

    @staticmethod
    def refine_favorite_child_edges(favorite_child_edge_floats, threshold=0.5):
        """Convert favorite child edge floats into integers.

        Simply transforms into integers by rounding them.

        TODO: Rounding can violate some of constraints.
              For example, one node may have more than one favorite child.
              Fix this issue.
        """
        _LOGGER.info('Favorite child round threshold: %s', str(threshold))
        return [round(value - threshold + 0.5)
                for value in favorite_child_edge_floats]

    def run(self):
        self.build_constraints()

        # [e1,e2,e3,...,n1,n2,n3,...,w]
        # objective function: minimize w
        objective = [0.0] * (self.num_edges +
                             self.num_nodes + 1)
        objective[-1] = 1.0
        objective = cvxopt.matrix(objective)

        LHS = cvxopt.spmatrix(*zip(*self.LHS_triples))
        RHS = cvxopt.matrix(self.RHS)

        _LOGGER.info('Start LP solver.')
        solution = cvxopt.solvers.lp(objective, LHS, RHS, solver='mosek')

        result = solution['x']
        _LOGGER.info(
            'LP solver finished. Relaxed makespan soultion: %f', result[-1])

        return self.refine_favorite_child_edges(result[:self.num_edges],
                                                threshold=self._threshold)


class DeviceWrapper(m_etf_v1.DeviceWrapper):

    def __init__(self, device_id, op_graph, device_graph, urgent_ts_ops):
        super(DeviceWrapper, self).__init__(device_id, device_graph, op_graph)
        self._urgent_ts_ops = urgent_ts_ops

    def get_next_ts(self):
        """Returns the timestamp when this device can have any action."""
        next_ts = None
        if self._state == DeviceState.AWAKE:
            # min(favorite child ready time, min. urgent ts)
            fc_op = self._op_graph.nodes[self._last_op['favorite']]
            fc_op_ready_tss = fc_op['ready_tss']
            if 'p' not in fc_op and self.id in fc_op_ready_tss:
                next_ts = fc_op_ready_tss[self.id]
            if len(self._urgent_ts_ops) > 0:
                next_urgent_ts = self._urgent_ts_ops[0].ts
                next_ts = min(next_ts or next_urgent_ts, next_urgent_ts)
            if next_ts is not None and next_ts <= self._current_ts:
                raise ValueError('Timestamp should move forward')
            return next_ts

        return super(DeviceWrapper, self).get_next_ts()

    def _get_earlist_op(self, ts_ops):
        """Find a op that has the earlist ready/urgent ts.

        If there are multiple ops that have the same ready/urgent tss,
        pick the one whose parent's device is not this op.
        This is for respecting favorite child.
        """
        # assumes ts_ops is sorted.
        earlist_ts_op = None
        for ts_op in ts_ops:
            if ts_op.op['memory'] > self.available_memory:
                continue

            if ts_op.ts > self._current_ts:
                # this op and after this are not ready to run
                break

            if earlist_ts_op is None:
                earlist_ts_op = ts_op
            else:
                if earlist_ts_op.ts < ts_op.ts:
                    break

                assert earlist_ts_op.ts == ts_op.ts
                parent_op_id = earlist_ts_op.op['parent']
                if parent_op_id != -1:
                    # the existing earlist op is a favorite child.
                    parent_op = self._op_graph.nodes[parent_op_id]
                    if parent_op['p'] != self.id:
                        # this op should be executed on another device
                        earlist_ts_op = ts_op

        if earlist_ts_op:
            earlist_ts_op.op['ready_ts'] = earlist_ts_op.ts
            return earlist_ts_op.op
        return None

    def _get_earlist_ready_op(self):
        return self._get_earlist_op(self._ready_ts_ops)

    def _get_earlist_urgent_op(self):
        return self._get_earlist_op(self._urgent_ts_ops)

    def _do_awake_action(self):
        urgent_op = self._get_earlist_urgent_op()
        if urgent_op:
            self._schedule_op(urgent_op)
            return urgent_op

        # wait until the last op's favorite child is ready
        fc_op_id = self._last_op['favorite']
        fc_op_ready_idx = placer_utils.find_index_of_ts_op_tuple(
            self._ready_ts_ops, fc_op_id)
        # TODO: handle when fc_op_ready_idx == 0,
        #       this can happen when the favorite child op ran
        #       on other devices
        fc_op_ready_ts = self._ready_ts_ops[fc_op_ready_idx]
        if fc_op_ready_ts.ts <= self._current_ts:
            fc_op_ready_ts.op['ready_ts'] = fc_op_ready_ts.ts
            self._schedule_op(fc_op_ready_ts.op)
            return fc_op_ready_ts.op

        return None

    def change_state(self):
        super(DeviceWrapper, self).change_state()
        if self._state == DeviceState.FREE and self._last_op is not None:
            # check whether there is a favorite child op that this device
            # can run earlist than other devices.
            # if so, set the state to AWAKE.
            fc_op_id = self._last_op['favorite']
            if fc_op_id != -1:
                fc_op = self._op_graph.nodes[fc_op_id]
                if 'p' not in fc_op and 'ready_tss' in fc_op:
                    # the favorite child does not run yet but is ready to run
                    fc_op_ready_tss = fc_op['ready_tss']
                    if self.id in fc_op_ready_tss:
                        min_ready_ts = min(fc_op_ready_tss.values())
                        if fc_op_ready_tss[self.id] == min_ready_ts:
                            self._state = DeviceState.AWAKE

    def schedule(self):
        assert len(self._ready_ts_ops) <= len(self._urgent_ts_ops)
        self.change_state()
        if self._state == DeviceState.AWAKE:
            return self._do_awake_action()

        return super(DeviceWrapper, self).schedule()


class SCT(m_etf_v1.ETF):

    def __init__(self, op_graph, device_graph):
        super(SCT, self).__init__(op_graph, device_graph)
        self.favorite_child_lp_solver = FavoriteChildLPSolver(op_graph)
        # TODO: urgent_ts might change on runtime due to the memory constraint.
        #       fix this.
        self._urgent_ts_ops = placer_utils.SortedTimestampOps()
        # replace devices
        self.devices = [
            DeviceWrapper(device_id, op_graph,
                          device_graph, self._urgent_ts_ops)
            for device_id in device_graph.nodes]

    def assign_favorite_child(self, favorite_child_edges):
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

    def _initialize_ops(self):
        initialized_ops = super(SCT, self)._initialize_ops()
        for initialized_op in initialized_ops:
            self._urgent_ts_ops.add_op(
                initialized_op['urgent_ts'], initialized_op)

    def _process_scheduled_op(self, scheduled_op):
        if scheduled_op is not None:
            super(SCT, self)._process_scheduled_op(scheduled_op)
            self._urgent_ts_ops.remove_op(scheduled_op)

    def _process_new_ready_ops(self, new_ready_ops):
        super(SCT, self)._process_new_ready_ops(new_ready_ops)
        for new_ready_op in new_ready_ops:
            self._urgent_ts_ops.add_op(new_ready_op['urgent_ts'], new_ready_op)

    def run(self):
        """Runs the SCT placement."""
        favorite_child_edges = self.favorite_child_lp_solver.run()
        self.assign_favorite_child(favorite_child_edges)
        runtime = self._run_schedule()
        _LOGGER.info('SCT estimated runtime: %f', runtime / 1e6)
        return runtime


def m_sct(op_graph, device_graph):
    sct = SCT(copy.deepcopy(op_graph), copy.deepcopy(device_graph))
    sct.run()
    placer_utils.transfer_placement(sct.op_graph, op_graph)
    return op_graph
