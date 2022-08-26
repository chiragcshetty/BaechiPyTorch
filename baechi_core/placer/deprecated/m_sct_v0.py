from cvxopt import spmatrix, matrix, solvers

from utils import logger

_LOGGER = logger.get_logger(__file__)


def to_float(arr):
    return [float(temp) for temp in arr]


def append_val(row_count, column, value,
               LHS_values, LHS_cindices, LHS_rindices):
    LHS_rindices.append(row_count)
    LHS_cindices.append(column)
    LHS_values.append(value)


def solve_favorite_child_lp(op_graph):
    """Solve the linear programming to calculate favorite children.

    LP variables are formatted in [e1,e2,e3,...,n1,n2,n3,...,w].
    e_i represents whether it is a favorite child edge
    n_i represents the start time of each op
    w represents makespan
    """

    num_nodes = op_graph.number_of_nodes()
    num_edges = op_graph.number_of_edges()
    _LOGGER.info('#nodes: %d, #edges: %d', num_nodes, num_edges)

    # Building LP constraints
    # a constraint is formatted in LHS <= RHS
    LHS_values = []
    # row indices. each two represents a constraint
    # the number of rows is equal to the number of contraints
    LHS_rindices = []
    # column indices. each column represents a variable
    LHS_cindices = []

    RHS = []  # right hand side values in constraints

    # rule 1: 0 <= x_i <= 1
    row_count = 0
    for i in range(num_edges):
        # x_i,j <= 1
        append_val(row_count, i, 1, LHS_values, LHS_cindices, LHS_rindices)
        RHS.append(1)
        row_count += 1

    for i in range(num_edges):
        # x_i,j >= 0
        append_val(row_count, i, -1, LHS_values, LHS_cindices, LHS_rindices)
        RHS.append(0)
        row_count += 1

    # s_k has a variable index, (num_edges) + i
    # rule 2: s_k >= 0
    for i in range(num_nodes):
        append_val(row_count, num_edges + i, -1,
                   LHS_values, LHS_cindices, LHS_rindices)
        RHS.append(0)
        row_count += 1

    # rule 3: s_i + p_i + c_i,j * x_i,j <= s_j
    for i, j in op_graph.edges:
        # s_i - s_j + c_i,j * x_i,j <= -p_i
        append_val(row_count, num_edges + i, 1,
                   LHS_values, LHS_cindices, LHS_rindices)
        append_val(row_count, num_edges + j, -1,
                   LHS_values, LHS_cindices, LHS_rindices)
        append_val(
            row_count, op_graph[i][j]['id'], op_graph[i][j]['weight'],
            LHS_values, LHS_cindices, LHS_rindices)
        RHS.append(-op_graph.nodes[i]['weight'])
        row_count += 1

    # rule 4:
    # for node i, node j s.t. there is an edge i -> j, \sum_j x_i,j >= |j| - 1
    for op_id in op_graph:
        if op_graph.out_degree[op_id] > 0:
            # \sum_j - x_i,j <= 1 - |j|
            for i, j in op_graph.out_edges(op_id):
                assert i == op_id
                append_val(row_count, op_graph[i][j]['id'], -1,
                           LHS_values, LHS_cindices, LHS_rindices)
            RHS.append(1 - op_graph.out_degree[op_id])
            row_count += 1

    # rule 5:
    # for node i, node j s.t. there is an edge j -> i, \sum_j x_j,i <= |j| - 1
    for op_id in op_graph:
        if op_graph.in_degree[op_id] > 0:
            # \sum_j -x_j,i <= 1 - |j|
            for i, j in op_graph.in_edges(op_id):
                assert j == op_id
                append_val(row_count, op_graph[i][j]['id'], -1,
                           LHS_values, LHS_cindices, LHS_rindices)
            RHS.append(1 - op_graph.in_degree[op_id])
            row_count += 1

    # rule 6: s_i + p_i <= w
    for op_id in op_graph:
        # s_i - w <= - p_i
        append_val(row_count, num_edges + op_id, 1,
                   LHS_values, LHS_cindices, LHS_rindices)
        # column (num_nodes + num_edges) represents the makespan variable
        append_val(row_count, num_nodes + num_edges, -1,
                   LHS_values, LHS_cindices, LHS_rindices)
        RHS.append(-op_graph.nodes[op_id]['weight'])
        row_count += 1

    # solve LP
    goal = [0] * (num_nodes + num_edges + 1)
    goal[-1] = 1
    goal = matrix(to_float(goal))
    # goal equation: w
    LHS = spmatrix(LHS_values, LHS_rindices, LHS_cindices)
    RHS = matrix(to_float(RHS))

    _LOGGER.info('Start LP solver.')
    res = solvers.lp(goal, LHS, RHS, solver='mosek')
    # res = msk.ilp(goal, LHS, RHS)

    solution = res['x']
    _LOGGER.info(
        'LP solver finished. Relaxed makespan soultion: %f', solution[-1])

    # transform float solution values to integers.
    for i in range(num_edges):
        if solution[i] < 0.5:
            solution[i] = 0
        else:
            solution[i] = 1

    return solution[:num_edges]


def m_sct(op_graph, device_graph):
    """Place operators on devices by using the m_sct algorithm.

    Solve LP to get favorite child and then do the scheduling.
    """
    # TODO: fix this
    memory_limit = device_graph.nodes[0]['memory_limit']

    # calculate favorite child
    favorite_child_list = solve_favorite_child_lp(op_graph)

    # init
    for _, op_data in op_graph.nodes.items():
        op_data['favorite'] = -1
        op_data['parent'] = 0

    for _, device_data in device_graph.nodes.items():
        device_data['l'] = -1
        device_data['s'] = 'free'

    for i, (op1_id, op2_id) in enumerate(op_graph.edges):
        if favorite_child_list[i] == 0:
            op1 = op_graph.nodes[op1_id]
            if op1['favorite'] != -1:
                _LOGGER.debug('Changing favorite child of op %d from %d to %d',
                              op1_id,
                              op1['favorite'],
                              op2_id)
            op1['favorite'] = op2_id

    # sets
    S = []
    R = [T for T in op_graph.nodes if op_graph.in_degree[T] == 0]
    ready = {}
    urgent = {}

    # main loop
    num_nodes = op_graph.number_of_nodes()
    time = 0
    while(len(S) != num_nodes):
        # calculate ready time and urgent time
        for T in R:
            for proc in device_graph.nodes:
                r_time = 0
                for pre in op_graph.in_edges(T):
                    pre = pre[0]
                    t = op_graph.nodes[pre]['t']
                    p = op_graph.nodes[pre]['weight']
                    c = op_graph[pre][T]['weight']
                    if op_graph.nodes[pre]['p'] == proc:
                        r_time = max(r_time, t + p)
                    else:
                        r_time = max(r_time, t + p + c)
                ready[(T, proc)] = r_time
        for T in R:
            u_time = 0
            for pre in op_graph.in_edges(T):
                pre = pre[0]
                t = op_graph.nodes[pre]['t']
                p = op_graph.nodes[pre]['weight']
                c = op_graph[pre][T]['weight']
                u_time = max(u_time, t + p + c)
            urgent[T] = u_time

        # calculate state of processor
        for proc in device_graph.nodes:
            p = device_graph.nodes[proc]
            if p['l'] != -1:
                T = p['l']
                if op_graph.nodes[T]['t'] + op_graph.nodes[T]['weight'] > time:
                    p['s'] = 'busy'
                elif op_graph.nodes[T]['favorite'] != -1:
                    favorite = op_graph.nodes[T]['favorite']
                    if favorite in R and ready[(favorite, proc)] < urgent[favorite]:
                        p['s'] = 'awake'
                    else:
                        p['s'] = 'free'
                else:
                    p['s'] = 'free'

        # schedule
        avail = []
        for proc in device_graph.nodes:
            p = device_graph.nodes[proc]
            # for free processor
            if p['s'] == 'free':
                for T in R:
                    if p['size'] + op_graph.nodes[T]['memory'] > memory_limit:
                        continue
                    if ready[(T, proc)] <= time:
                        op_graph.nodes[T]['t'] = time
                        op_graph.nodes[T]['p'] = proc
                        S.append(T)
                        R.remove(T)
                        p['l'] = T
                        p['size'] += op_graph.nodes[T]['memory']
                        avail.append(T)
                        break

            # for awake processor
            if p['s'] == 'awake':
                for T in R:
                    if p['size'] + op_graph.nodes[T]['memory'] > memory_limit:
                        continue
                    if urgent[T] <= time or (op_graph.nodes[p['l']]['favorite'] == T and ready[(T, proc)] <= time):
                        op_graph.nodes[T]['t'] = time
                        op_graph.nodes[T]['p'] = proc
                        S.append(T)
                        R.remove(T)
                        p['l'] = T
                        p['size'] += op_graph.nodes[T]['memory']
                        avail.append(T)
                        break

        # advance time
        new_t = float('Inf')
        for T in S:
            new_t_p = op_graph.nodes[T]['t'] + op_graph.nodes[T]['weight']
            if new_t_p > time and new_t_p < new_t:
                new_t = new_t_p
        for T in R:
            for proc in device_graph:
                new_t_p = ready[(T, proc)]
                if new_t_p > time and new_t_p < new_t:
                    new_t = new_t_p
        for T in R:
            new_t_p = urgent[T]
            if new_t_p > time and new_t_p < new_t:
                new_t = new_t_p
        time = new_t

        # add to R
        for T in avail:
            for suc in op_graph.neighbors(T):
                s = op_graph.nodes[suc]
                s['parent'] += 1
                if s['parent'] == op_graph.in_degree[suc]:
                    R.append(suc)

    # TODO: can be removed
    makespan = 0
    for T in op_graph.nodes():
        t = op_graph.nodes[T]['t'] + op_graph.nodes[T]['weight']
        makespan = max(makespan, t)
    _LOGGER.info('Makespan: %f sec', makespan / 1e6)

    return op_graph
