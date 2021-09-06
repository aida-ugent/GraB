import numpy as np
import networkx as nx

from cne_gradient.compute_embedding_gradient import node_gradient, node_hessian_inverse
from optimal_density_quantifier.local.compute_kde_gradient import gaussian_kde_custom


class LinkBenefitLocallyKDEGradientEmbeddingGradient():
    def __init__(self, cne, s1, s2, embeddings, graph, node_types_dict, src_density_metric_gradients=None, source_type='I',
                 destination_type='A',
                 target_density_type='U', src_hessian_dict=None):

        if src_hessian_dict is None:
            src_hessian_dict = dict()
        self.src_hessian_dict = src_hessian_dict

        self.cne = cne
        self.s1 = s1
        self.s2 = s2
        self.embeddings = embeddings

        adj_matrix = nx.to_scipy_sparse_matrix(graph, nodelist=sorted(graph.nodes))
        adj_matrix.eliminate_zeros()
        self.adj_matrix = adj_matrix

        # self.src_density_metric_gradients = src_density_metric_gradients
        if src_density_metric_gradients is not None:
            self.src_density_metric_dict = {int(i): src_density_metric_gradients[idx, :] for idx, i in enumerate(node_types_dict.get(source_type).get('node_ids'))}
        # self.src_density_metric_dict = self._compute_kde_grad(destination_type,
        #                                                  embeddings, node_types_dict,
        #                                                  source_type,
        #                                                  target_density_type)

    def get_probability(self, src_id):
        return self.cne.get_posterior_row(src_id)

    def get_benefit(self, src_id, dst_id):
        src_h_inverse = node_hessian_inverse(self.adj_matrix, src_id, self.cne.get_posterior_row(src_id), self.s1,
                                             self.s2, self.embeddings)
        src_gradient_wrt_link = node_gradient(src_h_inverse, self.embeddings, src_id, [dst_id], self.s1, self.s2)[0]
        src_kde_grad = self.src_density_metric_dict.get(int(src_id))
        benefit = np.dot(src_gradient_wrt_link, src_kde_grad)
        return src_gradient_wrt_link, src_kde_grad, benefit

    def get_benefits(self, src_id, dst_list):
        src_h_inverse = node_hessian_inverse(self.adj_matrix, src_id, self.cne.get_posterior_row(src_id), self.s1,
                                             self.s2, self.embeddings, node_hessian_val=self.src_hessian_dict.get(int(src_id)))
        src_gradient_wrt_links = node_gradient(src_h_inverse, self.embeddings, src_id, dst_list, self.s1, self.s2)
        src_kde_grad = self.src_density_metric_dict.get(int(src_id))
        benefits = np.dot(src_gradient_wrt_links, src_kde_grad)
        return src_gradient_wrt_links, src_kde_grad, benefits


def _find_best_candidates_greedy_locally(destination_node_ids, graph, source_node_ids,
                                         src_density_metrics, top_k, lbl):
    adj_matrix = nx.to_scipy_sparse_matrix(graph, nodelist=sorted(graph.nodes))
    adj_matrix.eliminate_zeros()
    benefit_dict = dict()
    for src_id, src_density_metric in zip(source_node_ids, src_density_metrics):
        dst_list = list()
        row_probs = lbl.get_probability(src_id)
        for dst_id in destination_node_ids:
            if adj_matrix[src_id, dst_id]:
                continue
            dst_list.append(dst_id)
        if dst_list:
            src_dst_metrics, src_kde_grad, benefits = lbl.get_benefits(src_id, dst_list)
            for benefit, dst_id, src_dst_metric in zip(benefits, dst_list, src_dst_metrics):
                benefit_dict[benefit] = (src_id, dst_id, src_density_metric, src_dst_metric)
    gradient_values = benefit_dict.keys()
    return_list = list()
    if gradient_values:
        for _ in range(top_k):
            max_grad = max(gradient_values)
            l = benefit_dict.pop(max_grad)
            # print(max_grad, l)
            if max_grad < 0:
                break
            return_list.append((l[0], l[1]))
    print("found top %d" % len(return_list))
    return return_list


def find_best_candidate_locally_by_embedding_gradient(cne, s1, s2, embeddings, graph, node_types_dict, kde_optimization_metric_calculator,
                                                      source_type='I',
                                                      destination_type='A',
                                                      target_density_type='U', top_k=100,
                                                      src_hessian_dict=None):
    destination_node_ids, source_node_ids, src_density_metric_gradients = _get_kde_optimization_metric(
        destination_type, embeddings, node_types_dict, source_type, target_density_type, kde_optimization_metric_calculator)
    lbl = LinkBenefitLocallyKDEGradientEmbeddingGradient(cne, s1, s2, embeddings, graph, node_types_dict, src_density_metric_gradients,
                                                   source_type=source_type,
                                                   destination_type=destination_type,
                                                   target_density_type=target_density_type,
                                                   src_hessian_dict=src_hessian_dict)
    return _find_best_candidates_greedy_locally(destination_node_ids, graph, source_node_ids,
                                         src_density_metric_gradients, top_k, lbl)


def _get_kde_optimization_metric(destination_type, embeddings, node_types_dict, source_type,
                                 target_density_type, kde_optimization_metric_calculator):
    source_node_ids = node_types_dict.get(source_type).get('node_ids', [])
    destination_node_ids = node_types_dict.get(destination_type).get('node_ids', [])
    target_node_ids = node_types_dict.get(target_density_type).get('node_ids', [])
    target_kde_obj = gaussian_kde_custom(embeddings[target_node_ids].T)
    # target_kde_values = target_kde_obj.evaluate(embeddings[source_node_ids].T)
    # target_kde_gradients = target_kde_obj.gradient(embeddings[source_node_ids].T)
    source_kde_obj = gaussian_kde_custom(embeddings[source_node_ids].T)
    # source_kde_values = source_kde_obj.evaluate(embeddings[source_node_ids].T)
    # source_kde_gradients = source_kde_obj.gradient(embeddings[source_node_ids].T)
    # n, d = embeddings.shape
    src_density_metric_gradients = kde_optimization_metric_calculator.get_metric_gradient(
        embeddings[source_node_ids], target_kde_obj, source_kde_obj)
    # src_density_metric_gradients = target_kde_gradients / np.tile(target_kde_values.reshape((-1, 1)),
    #                                                      (1, d)) - source_kde_gradients / np.tile(
    #     source_kde_values.reshape((-1, 1)), (1, d))
    return destination_node_ids, source_node_ids, src_density_metric_gradients
