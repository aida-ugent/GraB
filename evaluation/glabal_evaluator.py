import math

import numpy as np
import sklearn

from distance_calculation.emd_calculator import emd_calculator_optimized, emd_calculator_optimized_kmeans
from optimal_density_quantifier.local.compute_kde_gradient import gaussian_kde_custom


class GlobalEvaluator:
    @classmethod
    def evaluate(cls, embeddings, method, source_node_ids, target_node_ids,
                 emd=False):
        result = dict()
        if emd:
            result['emd'] = cls.emd(embeddings, method, source_node_ids, target_node_ids)
        return result

    @staticmethod
    def emd(embeddings, method, source_node_ids, target_node_ids):
        emd_new = emd_calculator_optimized_kmeans(embeddings[source_node_ids], embeddings[target_node_ids])
        print("*********************************************")
        print("emd: %.7f" % (emd_new))
        print("*********************************************")
        return emd_new
