import copy
import pickle

import numpy as np
import networkx as nx

from data_handler.data_common import get_graph
from embedding.cne_recommender.cne_debugged import ConditionalNetworkEmbedding
from embedding.cne_recommender.bg_dist_partite import BgDistBuilder


class NetworkEmbeddingMethod:
    def __init__(self, args, method='cne'):
        self.graph = None
        self.embeddings = None
        self.block_mask = None
        self.node_types_dict = None
        self.ne_method = None
        self.method = method
        self.args = args

    def apply_reembedding(self, graph, selected_links, max_iter=250):
        g = graph.copy()
        for l in selected_links:
            g.add_edge(l[0], l[1])
            g.add_edge(l[1], l[0])
        new_cne, embeddings_new = self.compute_embeddings_with_init_embeddings(
            g, recompute_prior=True, links_to_small_add=None, epsilon=0, max_iter=max_iter, ftol=1e-8, lr=0.01)
        return embeddings_new, g, new_cne

    def get_graph(self):
        return self.graph

    def get_embeddings(self):
        return self.embeddings

    def compute_embeddings_with_init_embeddings(self, g, recompute_prior=False, links_to_small_add=None,
                                                epsilon=0.0000001, max_iter=1, lr=0.001, ftol=1e-32, print_row_ids=None, use_newton=False
                                                # epsilon=0.1, max_iter=1, lr=0.01
                                                ):
        max_iter = int(max_iter)
        epsilon = float(epsilon)
        lr = float(lr)
        ftol = float(ftol)
        if self.method == 'cne':
            if recompute_prior:
                prior = None
            else:
                prior = self._get_prior(self.get_adj_matrix(self.graph))

            adj_matrix = self.get_adj_matrix(g)

            if links_to_small_add:
                adj_matrix = adj_matrix.astype(float)
                for l in links_to_small_add:
                    if adj_matrix[l[0], l[1]] > 0.5:
                        adj_matrix[l[0], l[1]] -= epsilon
                        adj_matrix[l[1], l[0]] -= epsilon
                    else:
                        adj_matrix[l[0], l[1]] += epsilon
                        adj_matrix[l[1], l[0]] += epsilon
            # cne_filename = self.get_cne_file_name(self.args.data_directory, self.args.method_pkl_file_name)
            # cne = self.load_cne(cne_filename)
            from copy import deepcopy
            cne = deepcopy(self.ne_method)
            cne, _ = self._cne_embedding(adj_matrix, init_embeddings=self.embeddings.copy(), cne=cne, max_iter=max_iter, lr=lr, prior=prior, ftol=ftol, print_row_ids=print_row_ids, use_newton=use_newton)
            return cne, cne.get_embeddings()

    def compute_embeddings(self):
        self.node_types_dict = self._initials()
        args = self.args

        if self.method == 'cne':
            cne = None
            cne_filename = self.get_cne_file_name(args.data_directory, args.method_pkl_file_name)
            print(cne_filename)
            if args.load_method_pkl:
                try:
                    cne = self.load_cne(cne_filename)
                except:
                    pass
            if cne is None:
                adj_matrix = self.get_adj_matrix(self.graph)
                cne, prior = self._cne_embedding(adj_matrix)
                # embeddings = cne.get_embeddings()
                if args.save_method_pkl:
                    self.save_cne(cne, cne_filename)
            self.embeddings = cne.get_embeddings()
            self.ne_method = cne
            complete_node_types_dict = self.complete_node_types_dict(self.node_types_dict)
            return cne, self.embeddings, self.graph, complete_node_types_dict
        return self.node_types_dict

    def complete_node_types_dict(self, node_types_dict):
        node_types_dict = copy.deepcopy(node_types_dict)
        args = self.args
        try:
            roles_filename = args.data_directory + "/types.csv"
            with open(roles_filename, 'r') as f:
                for line in f:
                    if '#' in line:
                        continue
                    line = line.strip('\n')
                    line_parts = line.split(',')
                    if len(line_parts) == 3:
                        start_idx, last_idx, node_type = line_parts
                        start_idx = int(start_idx)
                        last_idx = int(last_idx)
                        node_ids = [i for i in range(start_idx, last_idx + 1)]
                    else:
                        start_idx, node_type = line_parts
                        node_ids = [int(start_idx)]

                    if node_types_dict.get(node_type, None) is None:
                        node_types_dict[node_type] = {'node_ids': node_ids}
                    else:
                        node_types_dict[node_type]['node_ids'] += node_ids

        except Exception as e:
            print(e)
        return node_types_dict

    def get_cne_file_name(self, data_directory, method_pkl_file_name):
        cne_filename = data_directory + "/" + method_pkl_file_name
        return cne_filename

    def save_cne(self, cne, cne_filename):
        pickle.dump(cne, open(cne_filename, 'wb'))

    def load_cne(self, cne_filename):
        cne = pickle.load(open(cne_filename, 'rb'))
        return cne

    def _initials(self, csv_data_file_path="/data.csv"):

        args = self.args
        node_types_dict, block_start_last_idxs = self._get_nodes_types_dict(args)
        block_start_last_idxs.sort(key=lambda x: x[0])
        block_mask = []
        for i in range(0, len(block_start_last_idxs)):
            block_mask += [i for _ in range(block_start_last_idxs[i][0], block_start_last_idxs[i][1] + 1)]
        block_mask = np.array(block_mask)
        self.block_mask = block_mask
        input_graph_filename = args.data_directory + csv_data_file_path
        self.graph = get_graph(input_graph_filename, args.delimiter)
        return node_types_dict

    @staticmethod
    def _get_nodes_types_dict(args):
        block_start_last_idxs = []
        roles_filename = args.data_directory + "/roles.csv"
        node_types_dict = dict()
        with open(roles_filename, 'r') as f:
            for line in f:
                if '#' in line:
                    continue
                line = line.strip('\n')
                start_idx, last_idx, node_type = line.split(',')
                start_idx = int(start_idx)
                last_idx = int(last_idx)
                block_start_last_idxs.append((start_idx, last_idx))
                node_ids = [i for i in range(start_idx, last_idx + 1)]
                if node_types_dict.get(node_type, None) is None:
                    node_types_dict[node_type] = {'node_ids': node_ids}
                else:
                    node_types_dict[node_type]['node_ids'] += node_ids
        return node_types_dict, block_start_last_idxs

    def _cne_embedding(self, adj_matrix, init_embeddings=None, cne=None, max_iter=250, lr=0.1, prior=None, ftol=1e-16, print_row_ids=None, use_newton=False):
        if prior is None:
            prior = self._get_prior(adj_matrix)
        if not cne:
            cne = ConditionalNetworkEmbedding(adj_matrix, self.args.dimension, self.args.s1, self.args.s2, prior)
        else:
            cne.set_adj_matrix(adj_matrix)
        cne.fit(max_iter=max_iter, lr=lr, init_embeddings=init_embeddings, verbose=True, ftol=ftol)
        return cne, prior

    @staticmethod
    def get_adj_matrix(g):
        adj_matrix = nx.to_scipy_sparse_matrix(g, nodelist=sorted(g.nodes))
        adj_matrix.eliminate_zeros()
        return adj_matrix

    def _get_prior(self, adj_matrix):
        prior = BgDistBuilder.build(adj_matrix, self.args.prior, block_mask=self.block_mask)
        prior.fit()
        return prior
