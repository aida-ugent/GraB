import math
import random
import numpy as np
import networkx as nx
from ordered_set import OrderedSet

from cne_gradient.compute_embedding_gradient import node_hessian
from optimal_density_quantifier.local.compute_kde_gradient import gaussian_kde_custom
from optimal_density_quantifier.local.find_best_candidates import \
    find_best_candidate_locally_by_embedding_gradient


def find_best_candidates_random(
        args, kde_optimization_metric_calculator, network_embedding_method, cne, s1, s2, embeddings, graph,
        node_types_dict, source_type='I',
        destination_type='A', target_type='U', link_count=100,
        candidate_count=1000, max_iter=1,
        reembedding_link_count=None, multi_reembedding_batch_count=None, cne_max_iter=500):
    source_node_ids, source_old_kde_obj, src_hessian_dict, target_node_ids, target_old_kde_obj, destination_node_ids = _initial(
        cne, embeddings, graph, node_types_dict, s1, s2, source_type, target_type, destination_type,
        compute_src_hessian_dict=False)

    adj_mtx = network_embedding_method.get_adj_matrix(graph)
    all_possible_links = []
    for src_id in source_node_ids:
        for dst_id in destination_node_ids:
            if adj_mtx[src_id, dst_id] == 1:
                continue
            all_possible_links.append((src_id, dst_id))
    print(len(all_possible_links))
    if len(all_possible_links) > candidate_count:
        initial_candidate_link_list = random.sample(all_possible_links, k=candidate_count)
    else:
        initial_candidate_link_list = all_possible_links

    return _apply_greedy_alg(kde_optimization_metric_calculator, cne, embeddings, graph,
                             initial_candidate_link_list,
                             link_count, network_embedding_method, s1, s2,
                             source_node_ids, source_old_kde_obj, target_node_ids, target_old_kde_obj,
                             max_iter, reembedding_link_count, cne_max_iter)


def find_best_candidates_local_embedding_gradient(
        args, kde_optimization_metric_calculator, network_embedding_method, cne, s1, s2, embeddings, graph,
        node_types_dict, source_type='I',
        destination_type='A', target_type='U', link_count=100,
        candidate_count=1000, max_iter=None,
        reembedding_link_count=None, multi_reembedding_batch_count=None, cne_max_iter=500):
    if multi_reembedding_batch_count:
        reembedding_steps = int(link_count / multi_reembedding_batch_count)
    else:
        reembedding_steps = 1
    selected_links = list()
    link_count = int(link_count / reembedding_steps)
    candidate_count = int(candidate_count / reembedding_steps)
    if reembedding_link_count:
        reembedding_link_count = int(reembedding_link_count / reembedding_steps)
    step = 0
    for _ in range(reembedding_steps):
        if step > 0:
            embeddings, graph, cne = network_embedding_method.apply_reembedding(graph,
                                                                                current_selected_links,
                                                                                max_iter=cne_max_iter)
        source_node_ids, source_old_kde_obj, src_hessian_dict, target_node_ids, target_old_kde_obj, destination_node_ids = _initial(
            cne, embeddings, graph, node_types_dict, s1, s2, source_type, target_type, destination_type)

        initial_candidate_link_list = find_best_candidate_locally_by_embedding_gradient(
            cne, s1, s2, embeddings, graph, node_types_dict, kde_optimization_metric_calculator,
            source_type=source_type, destination_type=destination_type,
            target_density_type=target_type, top_k=candidate_count, src_hessian_dict=src_hessian_dict)

        current_selected_links = _apply_greedy_alg(kde_optimization_metric_calculator, cne, embeddings, graph,
                                                   initial_candidate_link_list,
                                                   link_count, network_embedding_method, s1, s2,
                                                   source_node_ids, source_old_kde_obj, target_node_ids,
                                                   target_old_kde_obj,
                                                   max_iter, reembedding_link_count, cne_max_iter)
        selected_links += current_selected_links
        step += 1
    return selected_links


def _apply_greedy_alg(kde_optimization_metric_calculator, cne, embeddings, graph, initial_candidate_link_list,
                      link_count,
                      network_embedding_method, s1, s2, source_node_ids,
                      source_old_kde_obj, target_node_ids, target_old_kde_obj, max_iter=None,
                      reembedding_link_count=None, cne_max_iter=500):
    if reembedding_link_count is None:
        reembedding_link_count = link_count
    initial_candidate_link_list = OrderedSet(initial_candidate_link_list)

    currently_selected_links = list()
    top_reserved_links = OrderedSet()
    iteration = 0
    not_selected_links = list()

    while len(currently_selected_links) < link_count and len(initial_candidate_link_list) > 0:
        top_reserved_links_list = list(top_reserved_links)
        top_reserved_links = OrderedSet()
        top_reserved_links2 = OrderedSet()
        for candidate_link in top_reserved_links_list + list(initial_candidate_link_list):
            try:
                initial_candidate_link_list.remove(candidate_link)
            except KeyError:
                pass  # do nothing!
            problemistic_flag = False
            for selected_link in currently_selected_links:
                if candidate_link[0] == selected_link[0] and not problemistic_flag:
                    top_reserved_links.append(candidate_link)
                    problemistic_flag = True
                    break

            if not problemistic_flag:
                currently_selected_links.append(candidate_link)
                if len(currently_selected_links) == reembedding_link_count:
                    break
        if len(currently_selected_links) > 0:
            if max_iter is not None and iteration >= max_iter:
                break
            else:
                iteration += 1
            embeddings_new, g, new_cne = network_embedding_method.apply_reembedding(graph,
                                                                                    currently_selected_links,
                                                                                    max_iter=cne_max_iter)

            source_new_kde_obj = gaussian_kde_custom(embeddings_new[source_node_ids].T)
            target_new_kde_obj = gaussian_kde_custom(embeddings_new[target_node_ids].T)

            new_currently_selected_links = list()
            not_selected_links = list()

            for l in currently_selected_links:
                node_id = l[0]
                node_new_embeddings = embeddings_new[node_id, :]
                node_old_embeddings = embeddings[node_id, :]
                new_density_metric = kde_optimization_metric_calculator.get_metric(
                    node_new_embeddings, target_new_kde_obj, source_new_kde_obj)[0]
                old_density_metric = kde_optimization_metric_calculator.get_metric(
                    node_old_embeddings, target_old_kde_obj, source_old_kde_obj)[0]
                if new_density_metric > old_density_metric:
                    new_currently_selected_links.append(l)
                else:
                    not_selected_links.append(l)

            currently_selected_links = new_currently_selected_links
    return currently_selected_links[:link_count]


def _initial(cne, embeddings, graph, node_types_dict, s1, s2, source_type, target_type, destination_type,
             compute_src_hessian_dict=True):
    source_node_ids = node_types_dict.get(source_type).get('node_ids', [])
    target_node_ids = node_types_dict.get(target_type).get('node_ids', [])
    destination_node_ids = node_types_dict.get(destination_type).get('node_ids', [])
    source_old_kde_obj = gaussian_kde_custom(embeddings[source_node_ids].T)
    target_old_kde_obj = gaussian_kde_custom(embeddings[target_node_ids].T)
    src_hessian_dict = dict()
    if compute_src_hessian_dict:
        adj_matrix = nx.to_scipy_sparse_matrix(graph, nodelist=sorted(graph.nodes))
        adj_matrix.eliminate_zeros()
        for src in source_node_ids:
            src = int(src)
            src_hessian_dict[src] = node_hessian(adj_matrix, src, cne.get_posterior_row(src), s1, s2, embeddings)
    return source_node_ids, source_old_kde_obj, src_hessian_dict, target_node_ids, target_old_kde_obj, destination_node_ids
