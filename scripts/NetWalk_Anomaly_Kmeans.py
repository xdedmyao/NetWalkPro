"""
    NetWalk (PyTorch) end-to-end with streaming K-Means anomaly scoring on edges.
    Embeds nodes per snapshot and produces per-snapshot anomaly CSVs.
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
sys.path.insert(0, str(SRC))

DATA_DIR = ROOT / "data" / "snapshots"
EMB_DIR = ROOT / "output" / "embeddings"
ANOM_DIR = ROOT / "output" / "anomalies"


from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter
import logging
import os
import glob
from datetime import datetime
import warnings

import numpy as np
import torch

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
# print("TF32 enabled:", torch.backends.cuda.matmul.allow_tf32)

import netwalk.Model as MD
from netwalk.netwalk_update import NetWalk_update
from anomaly.Kmeans_Anomaly import StreamingKMeansDetector

warnings.filterwarnings("ignore")

logger = logging.getLogger(__name__)
LOGFORMAT = "%(asctime).19s %(levelname)s %(filename)s: %(lineno)s %(message)s"


def print_time():
    return datetime.now().strftime('[INFO %Y-%m-%d %H:%M:%S]')


def embedding_code(model, data, n, args, snapshot_id, idx2node):
    """Fit model on one-hot walks and return (emb_array, emb_file_path).
    Save first column using original node ids to align with edge CSVs.
    """
    model.fit(data)
    node_onehot = np.eye(n)
    res = model.feedforward_autoencoder(node_onehot)

    ids = np.array([idx2node[i] for i in range(n)]).reshape(-1, 1)
    embeddings = np.concatenate((ids.reshape(-1, 1), res), axis=1)

    output_dir = Path(args.output_embeddings)
    output_dir.mkdir(parents=True, exist_ok=True)

    snapshot_file = output_dir / f"snapshot_{snapshot_id}.txt"
    os.makedirs(os.path.dirname(snapshot_file), exist_ok=True)
    fmt = ["%s"] + ["%g"] * res.shape[1]
    embeddings_to_save = np.column_stack((ids.astype(object), res.astype(float)))
    np.savetxt(snapshot_file, embeddings_to_save, fmt=fmt)

    print(f"{print_time()} Done! Embeddings saved: {snapshot_file}")
    return res, snapshot_file


def static_process(args):
    # Hyperparameters
    hidden_size = args.representation_size
    activation = torch.tanh
    rho = 0.05
    lamb = 0.00001
    beta = 0
    gama = 100
    walk_len = args.walk_length
    epoch = 200 if args.fast else 400
    batch_size = 2048
    learning_rate = 0.0005
    optimizer = "adam"
    corrupt_prob = 0

    # Data prep
    # normalize to absolute path to avoid relative '../' issues when invoked from other working dirs
    data_path = os.path.abspath(args.input)
    netwalk = NetWalk_update(data_path, walk_per_node=args.number_walks,
                             walk_len=args.walk_length, output_path=args.output_embeddings)
    n = len(netwalk.vertices)

    print(f"{print_time()} Number of nodes: {n}")
    print(f"{print_time()} Number of walks per node: {args.number_walks}")
    print(f"{print_time()} Walk len: {args.walk_length}")
    print(f"{print_time()} Starting embedding + anomaly pipeline...")

    dimension = [n, hidden_size]
    emb_model = MD.Model(activation, dimension, walk_len, n, gama, lamb, beta, rho,
                         epoch, batch_size, learning_rate, optimizer, corrupt_prob)

    detector = StreamingKMeansDetector(k=args.k, alpha=args.alpha, ratio_clip=args.ratio_clip)
    anomaly_dir = args.anomaly_dir
    edge_files = sorted(glob.glob(os.path.join(data_path, "*.csv")))
    snapshots = netwalk.snapshots
    max_snapshots = min(len(edge_files), len(snapshots))

    # Snapshot 0
    data = netwalk.getInitWalk()
    init_edges = snapshots[0]
    # skip re-processing snapshot 0 inside the streaming loop
    netwalk.idx = 1

    embedding, emb_file = embedding_code(emb_model, data, n, args, snapshot_id=0, idx2node=netwalk.idx2node)
    if edge_files:
        detector.process_month(emb_file, edge_files[0], args.output_anomalies)

    # Streaming snapshots
    snapshot_num = 1
    while netwalk.hasNext() and snapshot_num < max_snapshots:
        data = netwalk.nextOnehotWalks()
        if data is False:
            break
        embedding, emb_file = embedding_code(emb_model, data, n, args, snapshot_id=snapshot_num, idx2node=netwalk.idx2node)
        detector.process_month(emb_file, edge_files[snapshot_num], args.output_anomalies)
        snapshot_num += 1

    print("Pipeline finished.")


def main():
    parser = ArgumentParser("NETWALK_ANOMALY", formatter_class=ArgumentDefaultsHelpFormatter, conflict_handler='resolve')

    parser.add_argument('--input', nargs='?', default=DATA_DIR, help='Folder containing monthly CSV edge files')
    parser.add_argument('--output_embeddings', default=EMB_DIR, help='Output embedding prefix (per snapshot)')
    parser.add_argument('--output_anomalies', default=ANOM_DIR, help='Output embedding prefix (per snapshot)')
    parser.add_argument('--anomaly-dir', default=ANOM_DIR, help='Folder to save anomaly CSVs')
    parser.add_argument('--number_walks', default=20, type=int, help='Number of random walks per node')
    parser.add_argument('--walk-length', default=5, type=int, help='Length of each random walk')
    parser.add_argument('--representation-size', default=32, type=int, help='Embedding dimension')
    parser.add_argument('--seed', default=24, type=int, help='Random seed')
    parser.add_argument('--k', default=10, type=int, help='Number of clusters for streaming K-Means')
    parser.add_argument('--alpha', default=0.5, type=float, help='Decay factor for updating centers')
    parser.add_argument('--ratio-clip', default=4.0, type=float, help='Clip threshold for log ratio features')
    parser.add_argument('--topk', default=20, type=int, help='Top anomalies to print per snapshot')
    parser.add_argument('--fast', action='store_true', help='Use fewer epochs and smaller batch for quick runs')
    parser.add_argument('-l', '--log', dest='log', default='INFO', help='Log verbosity level')

    args = parser.parse_args()
    numeric_level = getattr(logging, args.log.upper(), None)
    logging.basicConfig(format=LOGFORMAT)
    logger.setLevel(numeric_level)

    static_process(args)


if __name__ == "__main__":
    main()
