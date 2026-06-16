import random
import numpy as np
import networkx as nx
from tqdm import tqdm
from scipy.sparse import coo_matrix
from scipy.sparse import csr_matrix
import os
import glob
import pandas as pd

class Reservoir:
    """
    Directed & Weighted reservoir sampling for monthly full graph snapshots
    """

    def __init__(self, edges, vertices, dim=20, seed=24):
        self.reservoir = {}
        self.degree = {}
        self.edge_weights = {}
        self.vertices = vertices
        self.reservoir_dim = dim
        self.seed = seed

        self.__build(edges)

    def __build(self, edges):
        """
        Construct reservoir using a full directed & weighted graph snapshot.
        edges format: [(u, v, w), ...]
        """
        g = nx.DiGraph()
        g.add_weighted_edges_from(edges)

        self.edge_weights = {(u, v): w for u, v, w in g.edges(data="weight")}
        np.random.seed(self.seed)
        for v in self.vertices:
            if v in g:
                nbrs = [(nbr, g[v][nbr]["weight"]) for nbr in g.successors(v)]

                if len(nbrs) == 0:
                    self.reservoir[v] = np.array([(None, 0)] * self.reservoir_dim)
                    self.degree[v] = 0
                    continue

                neighbors, weights = zip(*nbrs)
                weights = np.array(weights, dtype=float)
                prob = weights / weights.sum()

                # np.random.seed(self.seed)
                idx = np.random.choice(len(neighbors), size=self.reservoir_dim, p=prob)
                self.reservoir[v] = np.array([(neighbors[i], weights[i]) for i in idx])

                self.degree[v] = weights.sum()

            else:
                self.reservoir[v] = np.array([(None, 0)] * self.reservoir_dim)
                self.degree[v] = 0

    def update(self, new_edges):
        """
        Monthly rebuild: directly rebuild reservoir from new graph snapshot.
        new_edges must be a full edge list of the month.
        Format: [(u, v, w), ...]
        """
        self.__build(new_edges)




class WalkUpdate:
    """WalkUpdate: update the Reservoir and generate new batch of walks."""

    def __init__(self, init_edges, vertices, walk_len=3, walk_per_node=5, prev_percent=1, seed=24):
        self.init_edges = init_edges
        self.walk_len = walk_len
        self.walk_per_node = walk_per_node
        self.prev_percent = prev_percent
        self.seed = seed

        self.reservoir = Reservoir(edges=self.init_edges, vertices=vertices)

        self.prev_walks = self.__init_walks()

        self.new_walks = None
        self.training_walks = None

    def __init_walks(self):
        """
        Generate initial weighted random walks on a directed graph
        """
        g = nx.DiGraph()
        g.add_weighted_edges_from(self.init_edges)
        rand = random.Random(self.seed)
        walks = []
        nodes = list(g.nodes())

        for _ in range(self.walk_per_node):
            rand.shuffle(nodes)
            for node in nodes:
                walks.append(self.__random_walk(g, node, rand=rand))

        return walks

    def __random_walk(self, g, start, alpha=0, rand=random.Random(0)):
        """
        return a truncated weighted random walk on a directed graph
        :param alpha: probability of restarts
        :param start: the start node of the random walk
        """
        walk = [start]

        while len(walk) < self.walk_len:
            cur = walk[-1]
            neighbors = list(g.successors(cur))
            if not neighbors:
                break

            if rand.random() < alpha:
                walk.append(walk[0])
                continue

            weights = [g[cur][nbr].get('weight', 1) for nbr in neighbors]
            total = sum(weights)
            r = rand.random() * total

            cum = 0
            for nbr, w in zip(neighbors, weights):
                cum += w
                if cum >= r:
                    walk.append(nbr)
                    break

        return walk

    def __generate(self, new_edges, update_type="randomwalk"):
        walk_set = []
        rand = random.Random(self.seed)

        ## using random walk in the reservoir for updating new set of walks for training
        # it's slower but very accurate, it is probabilistically equal to do random walk
        # in the whole graph with all edges so far arrived
        if update_type == "randomwalk":
            start_nodes = set()
            for u, v, w in new_edges:
                start_nodes.add(u)
                start_nodes.add(v)

            for n in start_nodes:
                for _ in range(self.walk_per_node):
                    if len(self.reservoir.reservoir[n]) == 0:
                        continue

                    x_tuple = rand.choice(self.reservoir.reservoir[n])
                    x = x_tuple[0]
                    if x is None or len(self.reservoir.reservoir.get(x, [])) == 0:
                        continue

                    y_tuple = rand.choice(self.reservoir.reservoir[x])
                    y = y_tuple[0]
                    if y is None:
                        continue

                    walk_set.append([n, x, y])

            self.new_walks = walk_set
            self.training_walks = walk_set
            self.prev_walks = walk_set[:]

            # print("=====  snapshot walks =====")
            # for i, walk in enumerate(self.training_walks):
            #     print(f"walk {i}: {walk}")

            print("length of training walks:", len(self.training_walks))
            return

        assert self.walk_len < 5

        start_nodes = set()
        for u, v, w in new_edges:
            start_nodes.add(u)
            start_nodes.add(v)

        if self.walk_len == 3:
            for u, v, w in new_edges:
                for _ in range(self.walk_per_node):
                    # u - v - x
                    if len(self.reservoir.reservoir[v]) > 0:
                        x = rand.choice(self.reservoir.reservoir[v])
                        walk_set.append([u, v, x])

                    # v - u - x
                    if len(self.reservoir.reservoir[u]) > 0:
                        x = rand.choice(self.reservoir.reservoir[u])
                        walk_set.append([v, u, x])

        elif self.walk_len == 4:
            for u, v, w in new_edges:
                for _ in range(self.walk_per_node):

                    # u - v - x - y
                    if self.reservoir.reservoir[v]:
                        x = rand.choice(self.reservoir.reservoir[v])
                        if self.reservoir.reservoir[x]:
                            y = rand.choice(self.reservoir.reservoir[x])
                            walk_set.append([u, v, x, y])

                    # v - u - x - y
                    if self.reservoir.reservoir[u]:
                        x = rand.choice(self.reservoir.reservoir[u])
                        if self.reservoir.reservoir[x]:
                            y = rand.choice(self.reservoir.reservoir[x])
                            walk_set.append([v, u, x, y])

        self.new_walks = walk_set
        self.training_walks = walk_set
        self.prev_walks = walk_set[:]

    def update(self, new_edges):
        """
        Updating reservior and generate new set of walks for re-training using newly come edges
        :param new_edges: newly arrived edges
        :return: new set of training walks
        """
        # update reservior
        self.reservoir.update(new_edges)

        ## if reconduct randomwalk then the new set of walks are probabilistically equal to conducting
        ## randomwalk in the graph with all edges so far, it's slower than approximated method
        self.__generate(new_edges, update_type="randomwalk")

        return self.training_walks


class NetWalk_update:
    """
    Preparing both training initial graph walks and testing list of walks in each snapshot
    """

    def __init__(self, folder_path, walk_per_node=5, walk_len=3, output_path="../../output/embeddings/"):
        """
        Initialization of data preparing
        :param folder_path: folder containing monthly CSV edge files (format: u,v,w)
        :param walk_per_node: number of walks per node
        :param walk_len: length of each walk
        :param seed: random seed
        """
        self.folder_path = folder_path
        self.walk_len = walk_len
        self.walk_per_node = walk_per_node
        # self.seed = seed
        self.output_path = output_path
        self.idx = 0

        self.snapshots = self.__get_data_folder(folder_path)

        self.vertices = np.unique(np.concatenate(self.snapshots, axis=0)[:, :2])
        self.node2idx = {node: i for i, node in enumerate(self.vertices)}
        self.idx2node = {i: node for i, node in enumerate(self.vertices)}

        node_mapping_file = self.output_path / "node_mapping.txt"

        os.makedirs(os.path.dirname(node_mapping_file), exist_ok=True)

        with open(node_mapping_file, "w") as f:
            for idx, node in self.idx2node.items():
                f.write(f"{idx}\t{node}\n")

        init_edges = self.snapshots[0]
        self.walk_update = WalkUpdate(init_edges, self.vertices, walk_len=self.walk_len,
                                      walk_per_node=self.walk_per_node, seed=24)

    def __get_data_folder(self, folder_path):
        """
        Read all CSV files in folder and generate monthly snapshots
        CSV format: u,v,w
        """
        files = sorted(glob.glob(os.path.join(folder_path, "*.csv")))
        if not files:
            raise ValueError(f"No CSV files found in {folder_path}")
        snapshots = []
        for file in files:
            df = pd.read_csv(file, header=0)
            # Only use u, v, w columns; ignore extra coordinate columns
            edges = df[['u', 'v', 'w']].values
            snapshots.append(edges)
        print(f"Found {len(snapshots)} snapshots.")
        return snapshots

    def run(self):
        """
        Run NetWalk on all monthly snapshots
        """
        for edges in self.snapshots:
            training_set = self.walk_update.update(edges)
            onehot = self.getOnehot(training_set)
            # print(onehot)

    def getNumsnapshots(self):
        return len(self.snapshots)

    def nextOnehotWalks(self):
        if not self.hasNext():
            return False
        edges = self.snapshots[self.idx]
        self.idx += 1
        walks = self.walk_update.update(edges)
        return self.getOnehot(walks)

    def hasNext(self):
        return self.idx < len(self.snapshots)

    def getInitWalk(self):
        """
        Get initial walk list
        """
        walks = self.walk_update.prev_walks
        return self.getOnehot(walks)

    def getOnehot(self, walks):
        node_indices = []
        for walk in walks:
            for w in walk:
                if w in self.node2idx:
                    node_indices.append(self.node2idx[w])
                else:
                    raise ValueError(f"Node {w} not in node2idx mapping.")

        rows = np.array(node_indices, dtype=int)
        cols = np.arange(len(rows), dtype=int)
        data = np.ones(len(rows), dtype=int)

        coo = coo_matrix(
            (data, (rows, cols)),
            shape=(len(self.vertices), len(rows))
        )

        return coo.toarray()


if __name__ == "__main__":
    folder_path = "../../data/monthly_snapshots_sampled"
    netwalk = NetWalk_update(folder_path, walk_per_node=5, walk_len=3)
    netwalk.run()

