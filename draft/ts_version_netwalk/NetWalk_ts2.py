from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter
import logging
from datetime import datetime
import warnings
import tensorflow as tf
import numpy as np

import anomaly.Model_ts2 as MD  # 确保这是 TF2 版本的 Model
from netwalk.netwalk_update import NetWalk_update

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
    rho = 0.5
    lamb = 0.0017
    beta = 1
    gama = 340
    walk_len = args.walk_length
    epoch = 400
    batch_size = 20
    learning_rate = 0.1
    optimizer = "rmsprop"
    corrupt_prob = 0.0

    # STEP 1: Preparing data: folder of monthly CSV files
    data_path = args.input
    netwalk = NetWalk_update(data_path, walk_per_node=args.number_walks,
                             walk_len=args.walk_length, seed=args.seed)
    n = len(netwalk.vertices)

    print(f"{print_time()} Number of nodes: {n}")
    print(f"{print_time()} Number of walks: {args.number_walks}")
    print(f"{print_time()} Data size (walks*length): {args.number_walks * args.walk_length}")
    print(f"{print_time()} Generating network walks...")
    print(f"{print_time()} Clique embedding training...")

    dimension = [n, hidden_size]
    embModel = MD.Model(activation, dimension, walk_len, n, gama, lamb, beta, rho,
                        epoch, batch_size, learning_rate, optimizer, corrupt_prob)

    # Initial walks
    data = netwalk.getInitWalk()
    snapshots = netwalk.snapshots

    # STEP 2: Learning initial embeddings
    embedding_code(embModel, data, n, args)

    # STEP 3: Online updates over snapshots
    snapshotNum = 0
    while netwalk.hasNext():
        data = netwalk.nextOnehotWalks()
        snapshotNum += 1
        embedding_code(embModel, data, n, args)

    print("finished")


def embedding_code(model, data, n, args):
    """
    Feed 'data' (one-hot walks) into the embedding model and save the embeddings
    """
    global SNAPSHOT_ID

    # 训练模型
    model.fit(data)

    # 计算所有节点的 embedding
    node_onehot = np.eye(n, dtype=np.float32)
    embeddings = model.feedforward_autoencoder(node_onehot)

    ids = np.arange(n).reshape(-1, 1)
    embeddings = np.concatenate((ids, embeddings), axis=1)

    snapshot_file = f"{args.output}_snapshot_{SNAPSHOT_ID}.txt"
    np.savetxt(snapshot_file, embeddings, fmt="%g")

    print(f"{print_time()} Done! Embeddings are saved in \"{snapshot_file}\"")

    SNAPSHOT_ID += 1


def main():
    parser = ArgumentParser("NETWALK", formatter_class=ArgumentDefaultsHelpFormatter, conflict_handler='resolve')

    parser.add_argument('--input', nargs='?', default='../data/0.001', help='Folder containing monthly CSV edge files')
    parser.add_argument('--output', default='./tmp/embedding', help='Output representation file')
    parser.add_argument('--number_walks', default=5, type=int, help='Number of random walks per node')
    parser.add_argument('--walk-length', default=3, type=int, help='Length of each random walk')
    parser.add_argument('--representation-size', default=2, type=int, help='Number of latent dimensions')
    parser.add_argument('--seed', default=24, type=int, help='Random seed')
    parser.add_argument("-l", "--log", dest="log", default="INFO", help="Log verbosity level")

    args = parser.parse_args()
    numeric_level = getattr(logging, args.log.upper(), None)
    logging.basicConfig(format=LOGFORMAT)
    logger.setLevel(numeric_level)

    static_process(args)


if __name__ == "__main__":
    main()
