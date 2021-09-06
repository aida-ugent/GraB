import argparse

from embedding.embedding_computer import NetworkEmbeddingMethod
from evaluation.glabal_evaluator import GlobalEvaluator
from optimal_density_quantifier.local.methods import \
    find_best_candidates_local_embedding_gradient, find_best_candidates_random
from optimal_density_quantifier.local.optimization_metric import LogRatioMetric


def main(args):
    nem = NetworkEmbeddingMethod(args, method='cne')
    cne, embeddings, graph, node_types_dict = nem.compute_embeddings()
    source_node_ids = node_types_dict.get(args.source_type).get('node_ids', [])
    target_node_ids = node_types_dict.get(args.target_type).get('node_ids', [])

    GlobalEvaluator.evaluate(embeddings, cne, source_node_ids, target_node_ids,
                             emd=True)
    link_count = int(args.link_count)
    candidate_factor = float(args.candidate_factor)
    try:
        multi_reembedding_batch_count = int(args.reembedding_step_size)
    except:
        multi_reembedding_batch_count = None
    if args.method == 'SGraB' or args.method == 'Random':
        max_iter = 0
        candidate_factor = 1
    else:
        max_iter = 1
    if args.method == 'IRandom' or args.method == 'Random':
        func = find_best_candidates_random
    else:
        func = find_best_candidates_local_embedding_gradient
    reembedding_link_count = int(link_count * candidate_factor)
    _perform_method(
        args, candidate_factor, cne, embeddings, func, graph,
        link_count, max_iter, nem, node_types_dict,
        reembedding_link_count, source_node_ids, target_node_ids,
        multi_reembedding_batch_count)


def _perform_method(args, candidate_factor, cne, embeddings, func, graph,
                    link_count, max_iter, nem, node_types_dict, reembedding_link_count, source_node_ids,
                    target_node_ids, multi_reembedding_batch_count):
    cne_max_iter = int(args.max_iter)
    selected_links = func(args,
        LogRatioMetric, nem, cne, args.s1, args.s2, embeddings, graph,
        node_types_dict, link_count=link_count,
        candidate_count=int(link_count * candidate_factor * 100),
        max_iter=max_iter,
        reembedding_link_count=reembedding_link_count,
        source_type=args.source_type, destination_type=args.destination_type,
        target_type=args.target_type,
        multi_reembedding_batch_count=multi_reembedding_batch_count, cne_max_iter=cne_max_iter
    )
    if len(selected_links) < link_count:
        print("not_enough_links")
        return
    selected_links = selected_links[:link_count]

    embeddings_new, g, new_cne = nem.apply_reembedding(graph,
                                                       selected_links, max_iter=cne_max_iter)

    GlobalEvaluator.evaluate(embeddings_new, new_cne, source_node_ids, target_node_ids,
                             emd=True)

    return


def parse_args():
    parser = argparse.ArgumentParser(description=".")

    parser.add_argument('--data_directory', nargs='?',
                        default='data/toy_data_1',
                        help='Input folder of the graph path')
    parser.add_argument('--delimiter', default=',',
                        help='The delimiter used to separate the edgelist.')
    parser.add_argument('--dimension', type=int, default=2,
                        help='embedding dimension')
    parser.add_argument('--candidate_factor', type=str, default='3')
    parser.add_argument('--reembedding_step_size', type=str, default='')
    parser.add_argument('--k', type=float, default=100,
                        help='Sample size. Default is 100.')

    parser.add_argument('--s1', type=float, default=1,
                        help='Sigma 1. Default is 1.')

    parser.add_argument('--s2', type=float, default=2,
                        help='Sigma 2. Default is 2.')

    parser.add_argument('--prior', type=str, default='degree_per_block_eco',
                        help='cne prior: degree, degree_per_block, degree_per_block_eco, block or uniform')

    parser.add_argument('--save_method_pkl', type=int, default=1,
                        help='')

    parser.add_argument('--load_method_pkl', type=int, default=1,
                        help='')

    parser.add_argument('--method_pkl_file_name', type=str, default='cne2_degree_per_block_eco.pkl',
                        help='')

    parser.add_argument('--source_type', type=str, default='I',
                        help='')
    parser.add_argument('--target_type', type=str, default='U',
                        help='')
    parser.add_argument('--destination_type', type=str, default='A',
                        help='')
    parser.add_argument('--link_count', type=str, default='1')
    parser.add_argument('--method', type=str, default='GraB', help='GraB, SGraB, Random, IRandom')
    parser.add_argument('--max_iter', type=int, default=350,
                        help='max iter for CNE')
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    main(args)
