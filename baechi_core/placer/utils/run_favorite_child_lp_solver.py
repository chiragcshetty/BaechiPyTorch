from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import pickle
import tensorflow as tf
import functools

from placer import placer_lib
from placer.deprecated import m_sct_v0
from placer import m_sct

tf.app.flags.DEFINE_string('cost_path', '', 'cost path.')
tf.app.flags.DEFINE_boolean('use_old', False, 'use old version of SCT solver')

FLAGS = tf.app.flags.FLAGS


def build_op_graph(cost_path):
    with open(cost_path, 'rb') as f:
        cost_dict = pickle.load(f)

    if cost_dict['use_optimized_graph']:
        graphdef = cost_dict['optimized_graphdef']
    else:
        graphdef = cost_dict['graphdef']
    op_perfs = {op_perf.node: op_perf for op_perf in cost_dict['op_perfs']}

    tf_graph = tf.Graph()
    with tf_graph.as_default():
        tf.import_graph_def(graph_def=graphdef, name='')

    op_graph, op_index = placer_lib.create_placement_graph(
        tf_graph, op_perfs, placer_lib.get_comm_cost, is_op_perfs=True)

    return op_graph


def main(unparsed_args):
    if not FLAGS.cost_path:
        raise ValueError('cost_path is required')
    op_graph = build_op_graph(FLAGS.cost_path)

    if FLAGS.use_old:
        favorite_child_list = m_sct_v0.solve_favorite_child_lp(op_graph)
    else:
        solver = m_sct.FavoriteChildLPSolver(op_graph)
        favorite_child_list = solver.run()


if __name__ == "__main__":
    tf.app.run(main)
