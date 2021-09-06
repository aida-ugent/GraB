import numpy as np


def node_gradient(hi_inverse, x, i, k_list, s1, s2):
    gamma = 1 / s1 ** 2 - 1 / s2 ** 2
    return gamma * (x[k_list] - x[i]).dot(-hi_inverse)
    # return -hi_inverse.dot(x[k] - x[i])


def node_hessian_inverse(adj_matrix, i, posterior_prob_i_row, s1, s2, x, node_hessian_val=None):
    if node_hessian_val is None:
        node_hessian_val = node_hessian(adj_matrix, i, posterior_prob_i_row, s1, s2, x)
    h_i_inverse = np.linalg.pinv(node_hessian_val)

    return h_i_inverse


def node_hessian(adj_matrix, i, posterior_prob_i_row, s1, s2, x):
    gamma = 1 / s1 ** 2 - 1 / s2 ** 2
    h_i = fi_xi_gradient(adj_matrix, gamma, i, posterior_prob_i_row, x)
    return h_i


def fi_xi_gradient(adj_matrix, gamma, i, posterior_prob_i_row, x):
    n, d = x.shape
    h_i_part1 = np.identity(d) * np.sum(
        posterior_prob_i_row[[j for j in range(len(posterior_prob_i_row)) if j != i]] - adj_matrix[
            i, [j for j in range(len(posterior_prob_i_row)) if j != i]])
    x_i_diff = x[i, :] - x
    p_ij_2 = np.multiply(posterior_prob_i_row, 1 - posterior_prob_i_row)
    h_i_part2 = np.zeros((d, d))
    for j in range(n):
        if j == i:
            continue
        h_i_part2 += np.outer(x_i_diff[j], x_i_diff[j]) * p_ij_2[j]
    h_i_part2 *= gamma
    # h_i_part2 = gamma * (np.sum([np.outer(x_i_diff[j], x_i_diff[j])*p_ij_2[j] for j in range(n)], axis=0))
    h_i = gamma * (h_i_part1 - h_i_part2)
    return h_i
