import math

import networkx
import networkx as nx
import numpy as np
import torch
from global_params import plot_dir
import ray
import multiprocessing as mp


class ArrayGraph:
    """
    The Neuron Dependency Graph implementation

    Keeps an array of counts
    Makes a graph
    """

    def __init__(self, num_latent_predicates, alpha=20, labels=10, criterion=4):
        """
        Predicates must be falsifiable, but they do not need to be observed.
        :param num_pred:
        """
        self.num_latent_predicates = num_latent_predicates
        self.num_preds = labels + num_latent_predicates
        self.labels = labels
        # This is a triangular matrix
        # for element a, b (a<b), the contingency table is stored at self.count[a,b]
        # 4 indices: ab, anb, nab, nanb
        self.count = np.zeros((self.num_preds, self.num_preds, 4), dtype=np.long)
        self.constants = set()
        self.constant_threshold_p = 0.0001
        self.smallest_equiv = {}
        self.all_equivalent = {}
        self.alpha = alpha
        # assume independent
        self.null_p = np.zeros((self.num_preds, self.num_preds, 4), dtype=np.float32)
        self.l = np.zeros((self.num_preds, self.num_preds, 4), dtype=np.float32)
        self.robust_threshold = 1
        self.pool = None
        self.sample_size_4 = 0
        self.workers = 0

        # to color the graph
        # let's do targets first
        if labels:
            self.color_ranges = [(0, "red"), (labels, None)]
        else:
            self.color_ranges = [(0, None)]
        self.g: networkx.DiGraph = None

        self.criterion = criterion

    def load_count(self, count):
        self.count = count

    @property
    def sample_size(self):
        return int(self.sample_size_4 // 4)

    def vector_input(self, predicate_vector_array):
        """
        predicate_vector:
            predicate_vector = nodes >= 0.5
            predicate_vector = true_ones.long().cpu()
        """
        vec = predicate_vector_array
        for t in vec:
            f = 1 - t
            self.count[:, :, 0] += np.outer(t, t)
            self.count[:, :, 1] += np.outer(t, f)
            self.count[:, :, 2] += np.outer(f, t)
            self.count[:, :, 3] += np.outer(f, f)
        self.sample_size_4 += len(vec) / 4

    def parallel_vector_input(self, predicate_vector_array):
        @ray.remote
        class MatrixSum:
            def __init__(self, num_preds):
                self.count = np.zeros((num_preds, num_preds), dtype=np.int32)
                self.sample_size = 0

            def accumulate(self, a, b):
                self.count += np.outer(a, b)
                self.sample_size += 1
                return None

            def get(self):
                return self.count, self.sample_size

        if self.pool is None:
            ray.init()
            self.count = None
            batch_size = len(predicate_vector_array)
            workers = min(batch_size, mp.cpu_count() // 4)
            self.pool = [MatrixSum.remote(self.num_preds) for _ in range(workers * 4)]
            self.workers = workers

        for i, t in enumerate(predicate_vector_array):
            wi = i % self.workers
            f = 1 - t
            r = self.pool[4 * wi].accumulate.remote(t, t)
            self.pool[4 * wi + 1].accumulate.remote(t, f)
            self.pool[4 * wi + 2].accumulate.remote(f, t)
            self.pool[4 * wi + 3].accumulate.remote(f, f)
        return r

    def get_parallel_count(self):
        counts = ray.get([w.get.remote() for w in self.pool])
        self.count = np.zeros((self.num_preds, self.num_preds, 4), dtype=np.long)
        self.pool = None
        ray.shutdown()
        for i, res in enumerate(counts):
            c, ss = res
            self.count[:, :, i % 4] += c
            self.sample_size_4 += ss

    def propagate_all(self, node_set, g=None):
        """
        Given the set of nodes and the graph g, find the transitive closure of the nodes
        The returned nodes are nodes to be expected from this set

        :param node_set:
        :param g:
        :return: expected node set
        contra_count is expected to be 50% for null
        """
        # given the structure of the graph, return
        # get transitive closure of the set
        closure = set()
        queue = set(node_set)
        infr_count = 0
        contra_count = 0
        g = g or self.g
        while len(queue) != 0:
            u = queue.pop()
            closure.add(u)
            if u in g:
                for nbr in g[u]:
                    infr_count += 1
                    if nbr in queue or nbr in closure:
                        pass
                    else:
                        queue.add(nbr)

                    if nbr not in node_set:
                        contra_count += 1
        return closure, infr_count, contra_count

    def bidirectional_propagate_all(self, node_set, bidi_recall_thres, g=None, remove_contra=False):
        closure = set()
        queue = set(node_set)
        infr_count = 0
        g = g or self.g
        while len(queue) != 0:
            u = queue.pop()
            closure.add(u)
            if u in g:
                closure.update(self.get_all_equiv(u))
                succ = []
                for ue in [u] + self.get_all_equiv(u):
                    if ue in g:
                        ss = list(g[ue])
                        succ += ss
                        if bidi_recall_thres < 1:
                            bidi = g[negate(ue)]
                            for notv in bidi:
                                v = negate(notv)
                                if g[v][ue]["recall"] > bidi_recall_thres:
                                    succ.append(v)
                for nbr in succ:
                    infr_count += 1
                    if nbr in queue or nbr in closure:
                        pass
                    else:
                        queue.add(nbr)

        contra_count = 0
        for n in closure:
            if (not n[0], n[1]) in closure:
                contra_count += 1

        if remove_contra:
            removed = set()
            for n in closure:
                if (not n[0], n[1]) not in closure:
                    removed.add(n)
            if contra_count != 0:
                print(contra_count)
            return removed, infr_count, contra_count
        else:
            return closure, infr_count, contra_count

    def get_reachability_matrix(self, g, with_labels=False):
        labels = self.labels if with_labels else 0
        am = nx.to_numpy_array(g, nodelist=self.get_raw_nodes(labels))
        ts = torch.from_numpy(am).float()
        ts += torch.eye(ts.shape[0])
        return ts

    def find_all_constants(self):
        self.constants = set()
        for a in range(self.num_preds):
            pa = self.get_prior(a)
            if pa > 1 - self.constant_threshold_p:
                self.constants.add((True, a))
            elif pa < self.constant_threshold_p:
                self.constants.add((False, a))
        return self.constants

    def find_all_equivalence(self, verbose=False):

        smallest_equiv = {}

        all_equivalent = {(True, i): [] for i in range(self.num_preds)}
        all_equivalent.update({(False, i): [] for i in range(self.num_preds)})

        for a in range(self.num_preds):
            for b in range(a + 1, self.num_preds):
                if self.equiv((True, a), (True, b), verbose):
                    all_equivalent[(True, a)].append((True, b))
                    all_equivalent[(True, b)].append((True, a))
                    if (True, b) not in smallest_equiv:
                        smallest_equiv[(True, b)] = (True, a)
                if self.equiv((True, a), (False, b), verbose):
                    all_equivalent[(True, a)].append((False, b))
                    all_equivalent[(True, b)].append((False, a))
                    if (True, b) not in smallest_equiv:
                        smallest_equiv[(True, b)] = (False, a)

        self.smallest_equiv = smallest_equiv
        self.all_equivalent = all_equivalent
        return self.smallest_equiv

    def equiv(self, a, b, verbose=False):
        return self.implies(a, b, verbose) and self.implies((not a[0], a[1]), (not b[0], b[1]), verbose)

    def is_constant(self, a):
        return (True, a) in self.constants or (False, a) in self.constants

    def get_graph(self, with_negation=True, verbose=False):
        """

        :return:
        """
        print(f"Getting graph with {self.alpha=}")
        g = nx.DiGraph()
        self.compute_null_p()
        self.find_all_equivalence()
        self.find_all_constants()

        # get edges
        edges = set()
        for a in range(self.num_preds):
            for b in range(max(self.labels, a + 1), self.num_preds):
                ee = [((True, a), (True, b)), ((True, a), (False, b)),
                      ((False, a), (True, b)), ((False, a), (False, b))]
                for e in ee:
                    p, q = e
                    if self.is_constant(p[1]) or self.is_constant(q[1]):
                        continue
                    if self.equiv(p, q, False):
                        continue
                    if self.implies(p, q, verbose):
                        edges.add(e)

        if with_negation:
            negation_set = set()
            for e in edges:
                # e.g. ((True, 3), (False, 4)) becomes ((True, 4), (False, 3))
                not_e = ((not e[1][0], e[1][1]), (not e[0][0], e[0][1]))
                negation_set.add(not_e)
            edges = edges.union(negation_set)

        g.add_edges_from(edges)

        g.add_nodes_from(self.get_raw_nodes())

        self.g = g
        self.compute_edge_precision_recall(g)
        self.compute_prior(g)
        return g

    def compute_prior(self, g):
        for node in g.nodes:
            p = self.get_prior(node)
            g.nodes[node]["prior"] = p

    def compute_edge_precision_recall(self, g):
        for edge in g.edges:
            # high precision is expected. recall is arbitrary
            g.edges[edge]["precision"] = self.get_precision(edge[0], edge[1])
            g.edges[edge]["recall"] = self.get_recall(edge[0], edge[1])

    def get_precision(self, a, b):
        idx = 0
        if not b[0]:
            idx += 1
        if not a[0]:
            idx += 2
        a_true_b_true = self.count[a[1], b[1], idx]
        a_true = self.count[a[1], a[1], 0 if a[0] else 3]
        precision = a_true_b_true / a_true
        return precision

    def get_recall(self, a, b):
        idx = 0
        if not b[0]:
            idx += 1
        if not a[0]:
            idx += 2
        a_true_b_true = self.count[a[1], b[1], idx]
        b_true = self.count[b[1], b[1], 0 if b[0] else 3]
        recall = a_true_b_true / b_true
        return recall

    def get_equiv(self, node):
        if (True, node[1]) in self.smallest_equiv:
            if node[0]:
                return self.smallest_equiv[node]
            else:
                e = self.smallest_equiv[(True, node[1])]
                return (not e[0], e[1])
        else:
            return None

    def get_all_equiv(self, node):
        if node[0]:
            nodes = self.all_equivalent[node]
        else:
            nodes = self.all_equivalent[(True, node[1])]
            nodes = [(not n[0], n[1]) for n in nodes]
        return nodes

    def get_raw_nodes(self, label=0):
        nodes = [(True, a) for a in range(label, self.num_preds)]
        nodes += [(False, a) for a in range(label, self.num_preds)]
        return nodes

    def compute_null_p(self):
        # compute null_p
        for a in range(self.num_preds):
            pa = self.get_prior(a)
            for b in range(a + 1, self.num_preds):
                pb = self.get_prior(b)
                self.null_p[a][b][0] = pa * pb
                self.null_p[a][b][1] = pa * (1 - pb)
                self.null_p[a][b][2] = (1 - pa) * pb
                self.null_p[a][b][3] = (1 - pa) * (1 - pb)

    def get_prior(self, a):
        if isinstance(a, int):
            ca = self.count[a][a][0]
            cnota = self.count[a][a][3]
            pa = ca / (ca + cnota)
            return pa
        else:
            # a=(True, 32) e.g.
            ca = self.count[a[1]][a[1]][0 if a[0] else 3]
            cnota = self.count[a[1]][a[1]][3 if a[0] else 0]
            pa = ca / (ca + cnota)
            return pa

    def get_all_prior(self):
        return [self.get_prior(i) for i in range(self.num_preds)]

    def nice_plot(self, g=None, name=plot_dir / "dot_render.png", save='png', rename=True, omit_label=False,
                  omit_equiv=False):
        g = g or self.g
        if self.labels:
            # !!!! remove edges between labels
            to_remove = set()
            for e in g.edges:
                p, q = e
                if p[1] < self.labels and q[1] < self.labels:
                    to_remove.add(e)
            g.remove_edges_from(to_remove)

        if omit_label:
            n = range(self.labels)
            to_remove = set()
            for i in n:
                to_remove.add((True, i))
                to_remove.add((False, i))
            g.remove_nodes_from(to_remove)

        if omit_equiv:
            n = range(self.num_preds)
            to_remove = set()
            for i in n:
                if (True, i) in self.smallest_equiv:
                    to_remove.add((True, i))
                    to_remove.add((False, i))

        g.remove_nodes_from(list(nx.isolates(g)))

        f = nx.algorithms.dag.transitive_reduction(g)
        transfer_attributes(g, f)
        if rename:
            renaming = {}
            for v in f:
                new_name = f"{v[1]}" if v[0] else f"n{v[1]}"
                renaming[v] = new_name
            f = nx.relabel_nodes(f, renaming)
            nx.set_node_attributes(f, renaming, "id")

        # color the nodes
        for seg_idx, (start_idx, color) in enumerate(self.color_ranges):
            if seg_idx != len(self.color_ranges) - 1:
                end_idx, _ = self.color_ranges[seg_idx + 1]
            else:
                end_idx = self.num_preds

            r = range(start_idx, end_idx)
            for i in r:
                for node_name in ("n" + str(i), str(i)):
                    if color is not None:
                        try:
                            f.nodes[node_name]["color"] = color
                        except KeyError:
                            pass

        dot = nx.nx_pydot.to_pydot(f)

        for node in dot.get_nodes():
            node.set_id(node.get_name())

        dot.set_rankdir("BT")
        if save is True:
            save = "png"
        dot.write(name, format=save)
        print(f"Plot written to {name}")
        return dot, f

    def implies(self, p_name, q_name, verbose=True):
        p, q = p_name[1], q_name[1]
        assert p < q
        pq_arr = self.count[p, q]
        all_observed = pq_arr.sum()
        # count_idx = 0 if p_name[0] else 2
        # count_idx += 0 if q_name[0] else 1

        zero_idx = 0 if p_name[0] else 2
        zero_idx += 1 if q_name[0] else 0

        null_l = self.null_p[p, q, zero_idx]
        null_l = null_l / (1 - null_l)
        null_count = self.null_p[p, q, zero_idx] * all_observed

        signi_p = self.null_p[p, q, zero_idx] / self.alpha
        signi_l = signi_p / (1 - signi_p)
        signi_count = signi_p * all_observed
        real_count = self.count[p, q, zero_idx]
        if self.criterion == 1:
            if real_count <= signi_count:
                if verbose:
                    print(f"{str(p_name):12}->{str(q_name):12} with {self.count[p, q, zero_idx]:3} counts, "
                          f"{null_l=:.5f}, {null_count=:.5f}, {signi_l=:.5f}, {signi_count=:4.1f}")
                return True
            else:
                return False
        elif self.criterion == 4:
            # criterion in paper.
            if real_count <= signi_count and signi_count > self.robust_threshold:
                if verbose:
                    print(f"{str(p_name):12}->{str(q_name):12} with {self.count[p, q, zero_idx]:3} counts, "
                          f"{null_l=:.5f}, {null_count=:.5f}, {signi_l=:.5f}, {signi_count=:4.1f}")
                return True
            else:
                return False
        else:
            raise

    def save_count(self, name):
        import pickle
        pickle.dump(self.count, open(name, "wb"))

    def load_count_pkl(self, name):
        import pickle
        count = pickle.load(open(name, "rb"))
        self.load_count(count)


def tensor_to_nodes(tensor):
    tensor = tensor > 0.5
    nodes = []
    for i, truth in enumerate(tensor):
        nodes.append((truth.item(), i))
    return nodes


def transfer_attributes(g, f):
    # get all attributes
    attrs = set()
    for k, v in g.nodes.items():
        for at in v:
            if at not in attrs:
                attrs.add(at)
    for at in attrs:
        di = nx.get_node_attributes(g, at)
        nx.set_node_attributes(f, di, at)
        pass
    return f


def negate(node):
    return (not node[0], node[1])
