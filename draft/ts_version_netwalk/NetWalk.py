"""
    Created on: 2018-12-24
    License: BSD 3 clause

    Copyright (C) 2018
    Author: Wei Cheng <weicheng@nec-labs.com> & Wenchao Yu
    Affiliation: NEC Labs America
"""
from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter
import logging
from datetime import datetime
import warnings
import tensorflow as tf
import numpy as np
import networkx as nx

# from framework.imports import *
import anomaly.Model as MD
from netwalk.netwalk_update import NetWalk_update
# from plots.viz_stream import viz_stream

SNAPSHOT_ID = 0
warnings.filterwarnings("ignore")
logger = logging.getLogger(__name__)
LOGFORMAT = "%(asctime).19s %(levelname)s %(filename)s: %(lineno)s %(message)s"


def print_time():
    return datetime.now().strftime('[INFO %Y-%m-%d %H:%M:%S]')


def static_process(args):
    # STEP 0: Parameters
    hidden_size = args.representation_size
    activation = tf.nn.sigmoid
    rho = 0.05  # sparsity ratio
    lamb = 0.0005  # weight decay
    beta = 1  # sparsity weight
    gama = 1  # autoencoder weight
    walk_len = args.walk_length
    epoch = 400
    batch_size = 64  # number of epoch for optimizing, could be larger
    learning_rate = 0.001  # learning rate, for adam, using 0.01, for rmsprop using 0.1
    optimizer = "rmsprop"
    corrupt_prob = [0]

    # STEP 1: Preparing data: folder of monthly CSV files
    data_path = args.input
    netwalk = NetWalk_update(data_path, walk_per_node=args.number_walks,
                             walk_len=args.walk_length, output_path=args.output)
    n = len(netwalk.vertices)

    print("{} Number of nodes: {}".format(print_time(), n))
    print("{} Number of walks: {}".format(print_time(), args.number_walks))
    print("{} Data size (walks*length): {}".format(print_time(), args.number_walks * args.walk_length))
    print("{} Generating network walks...".format(print_time()))
    print("{} Clique embedding training...".format(print_time()))

    dimension = [n, hidden_size]
    embModel = MD.Model(activation, dimension, walk_len, n, gama, lamb, beta, rho,
                        epoch, batch_size, learning_rate, optimizer, corrupt_prob)

    # Initial walks
    data = netwalk.getInitWalk()
    snapshots = netwalk.snapshots

    init_edges = snapshots[0]
    G = nx.DiGraph()
    for u, v, w in init_edges:
        G.add_edge(u, v, weight=float(w))
    edge_list = G.edges()
    tuples = tuple(map(tuple, init_edges[:, :2]))
    rm_list = [x for x in edge_list if x not in tuples]
    # fig = plt.figure(figsize=(12, 12))
    # viz_stream(rm_list, fig, 5, 2, 1)

    # STEP 2: Learning initial embeddings
    embedding_code(embModel, data, n, args)

    # STEP 3: Online updates over snapshots
    snapshotNum = 0
    while netwalk.hasNext():
        data = netwalk.nextOnehotWalks()
        current_edges = snapshots[snapshotNum]
        tuples = tuple(map(tuple, current_edges[:, :2])) + tuples
        snapshotNum += 1
        embedding_code(embModel, data, n, args)

        G = nx.DiGraph()
        for u, v, w in current_edges:
            G.add_edge(u, v, weight=float(w))
        edge_list = G.edges()
        rm_list = [x for x in edge_list if x not in tuples]
        # viz_stream(rm_list, fig, 5, 2, snapshotNum + 1)

    # plt.show()
    print("finished")


def embedding_code(model, data, n, args):
    """
    Feed 'data' (one-hot walks) into the embedding model and save the embeddings
    """
    global SNAPSHOT_ID

    model.fit(data)
    node_onehot = np.eye(n)
    res = model.feedforward_autoencoder(node_onehot)
    ids = np.transpose(np.array(range(n)))
    ids = np.expand_dims(ids, axis=1)
    embeddings = np.concatenate((ids, res), axis=1)
    # np.savetxt(args.output, embeddings, fmt="%g")

    snapshot_file = f"{args.output}_snapshot_{SNAPSHOT_ID}.txt"
    np.savetxt(snapshot_file, embeddings, fmt="%g")

    print("{} Done! Embeddings are saved in \"{}\"".format(print_time(), args.output))

    SNAPSHOT_ID += 1


def main():
    parser = ArgumentParser("NETWALK", formatter_class=ArgumentDefaultsHelpFormatter, conflict_handler='resolve')

    parser.add_argument('--input', nargs='?', default='../data/0.001', help='Folder containing monthly CSV edge files')
    parser.add_argument('--output', default='./tmp/embedding', help='Output representation file')
    parser.add_argument('--number_walks', default=5, type=int, help='Number of random walks per node')
    parser.add_argument('--walk-length', default=5, type=int, help='Length of each random walk')
    parser.add_argument('--representation-size', default=8, type=int, help='Number of latent dimensions')
    parser.add_argument('--seed', default=24, type=int, help='Random seed')
    parser.add_argument("-l", "--log", dest="log", default="INFO", help="Log verbosity level")

    args = parser.parse_args()
    numeric_level = getattr(logging, args.log.upper(), None)
    logging.basicConfig(format=LOGFORMAT)
    logger.setLevel(numeric_level)

    static_process(args)


if __name__ == "__main__":
    main()
