import time
import numpy as np
import pandas as pd
from scipy import sparse

from embedding.cne.degree_prior import BackgroundDistribution


class BgDistBuilder:
    @staticmethod
    def build(A, prior, block_mask=None, attributes=None):
        """
        :param A: adjacency matrix, preferably CSR (compressed sparse row) matrix.
        :param prior: prior type.
        :param block_mask: indicates which block number each node in A belongs to.
        :param attributes: a list of |attribute_types| arrays that map nodes to the corresponding attribute values.
        Column sums will taken over nodes with identical attribute values.
        :return: the desired background distribution object (unfitted).
        """
        if prior == 'degree' or prior == 'uniform':
            return BackgroundDistribution(A, prior)

        if prior == 'block':
            return BgDistBlock(A, block_mask)

        if prior == 'degree_per_block':
            return BgDistDegreeBlock(A, block_mask, attributes)

        if prior == 'degree_per_block_eco':
            from embedding.cne_recommender.bg_dist_partite_economical import BgDistDegreeBlockEco
            return BgDistDegreeBlockEco(A, block_mask)
        if prior == 'degree_per_block_eco_mem':
            from embedding.cne_recommender.bg_dist_partite_economical_memory_efficient import BgDistDegreeBlockEcoMemoryEfficient
            return BgDistDegreeBlockEcoMemoryEfficient(A, block_mask)

        raise ValueError('Prior {:s} is not implemented'.format(prior))


class BgDistBlockAbstract:
    def __init__(self, A, block_mask):
        self._A = sparse.csr_matrix(A)
        self._block_mask = block_mask

    def fit(self):
        """
        Find the maximum entropy distribution, subject to the given constraints.
        """
        raise NotImplementedError

    def get_probability(self, row_ids, col_ids):
        """
        Compute P probability for the elements specified by row_ids and col_ids.
        :param row_ids: For every entry, its row index.
        :param col_ids: For every entry, its col index.
        :return: the probabilities of the specified elements in the fitted distribution.
        """
        raise NotImplementedError

    def get_full_P_matrix(self):
        pass

    def set_full_P_matrix(self, P):
        pass


class BgDistBlock(BgDistBlockAbstract):
    # Constraint(s): total block sum.

    def __init__(self, A, block_mask):
        super(BgDistBlock, self).__init__(A, block_mask)

        # Count block sizes and number of blocks.
        part_nbs, self._part_counts = np.unique(self._block_mask, return_counts=True)
        self._nb_parts = part_nbs.shape[0]
        self.__part_density = np.zeros((self._nb_parts, self._nb_parts))

    def fit(self):
        for i in range(self._nb_parts):
            for j in range(self._nb_parts):
                # Collect all the rows from block i.
                rows = self._A[self._block_mask == i]

                # From those rows, collect the elements from block j.
                subA = rows[:, self._block_mask == j]

                n, m = subA.shape
                self.__part_density[i, j] = subA.count_nonzero() / (n * m)

    def get_probability(self, row_ids, col_ids):
        row_parts = self._block_mask[row_ids]
        col_parts = self._block_mask[col_ids]

        return self.__part_density[row_parts, col_parts]

    def get_row_probability(self, row_ids, col_ids):
        return self.get_probability(row_ids, col_ids)


class BgDistDegreeBlock(BgDistBlockAbstract):
    # Constraint(s): total block sum and total node degree.

    def __init__(self, A, block_mask, attributes=None):
        super(BgDistDegreeBlock, self).__init__(A, block_mask)
        self.__atttributes = attributes

        self.__P = None

    def fit(self):
        """
        For now, this function was made as a basic example for priors that use attribute-wise sums. In the future,
        it should also be possible to extend to economical lambdas.
        Assumptions:
        - The network is undirected. (the whole matrix is computed, but the topright triangle is mirrored to the
        bottomleft at the end)
        - Attributes are only defined for rows, not columns. The sums are then taken over the columns.
        """

        block_types = np.unique(self._block_mask)
        assert np.all(block_types == np.arange(block_types.shape[0]))

        P = np.empty(self._A.shape, dtype=np.float)
        P[:] = np.nan

        for type_i in range(block_types.shape[0]):
            row_mask = self._block_mask == type_i
            for type_j in range(type_i, block_types.shape[0]):
                col_mask = self._block_mask == type_j

                sub_A = self._A[np.ix_(row_mask, col_mask)]

                # First, check if the total sum is not just zero. In that case, we can avoid doing any calculations.
                total_sum = sub_A.count_nonzero()
                if total_sum == 0:
                    P[np.ix_(row_mask, col_mask)] = 0.0
                    P[np.ix_(col_mask, row_mask)] = 0.0
                    continue

                # Lamdas aggregator keeps track of several 'lambdas' objects.
                lambdas = LambdasAggregator()

                # Define some functions to easily construct row and column lambdas.
                def construct_row_lambdas(row_mask_, col_mask_):
                    sub_sub_A = sub_A[np.ix_(row_mask_, col_mask_)]

                    # Compute the row sum for the given submatrix.
                    # The expected row sum will have to match the actual sum.
                    row_sums = sub_sub_A.sum(axis=1).A.squeeze()

                    # Construct Lambdas object for the row degree prior.
                    row_lambdas = RowDegreeLambdas(row_sums, row_mask_, col_mask_)
                    lambdas.add_lambdas_object(row_lambdas)

                def construct_col_lambdas(row_mask_, col_mask_):
                    sub_sub_A = sub_A[np.ix_(row_mask_, col_mask_)]

                    # The col_sums are computed in a similar way.
                    col_sums = sub_sub_A.sum(axis=0).A.squeeze()

                    # Construct Lambdas object for the column degree prior.
                    col_lambdas = ColumnDegreeLambdas(col_sums, row_mask_, col_mask_)
                    lambdas.add_lambdas_object(col_lambdas)

                # Where the row and col masks are true, we build sub-masks of all ones.
                sub_full_row_mask = np.ones(np.sum(row_mask), dtype=np.bool)
                sub_full_col_mask = np.ones(np.sum(col_mask), dtype=np.bool)

                if self.__atttributes is None:
                    construct_row_lambdas(sub_full_row_mask, sub_full_col_mask)
                    construct_col_lambdas(sub_full_row_mask, sub_full_col_mask)

                else:
                    already_a_full_row_constraint = False
                    already_a_full_col_constraint = False
                    for attribute_array in self.__atttributes.values():
                        possible_values = pd.unique(attribute_array)
                        for attribute_val in possible_values:
                            if attribute_val == "N/A":
                                continue

                            sub_row_mask = attribute_array[row_mask] == attribute_val
                            # If there are no attributed rows to sum over for the column lambda, then make sure that
                            # there is a constraint that sums over all rows.
                            if not np.any(sub_row_mask):
                                if not already_a_full_col_constraint:
                                    construct_col_lambdas(sub_full_row_mask, sub_full_col_mask)
                                    already_a_full_col_constraint = True
                            else:
                                construct_col_lambdas(sub_row_mask, sub_full_col_mask)

                            sub_col_mask = attribute_array[col_mask] == attribute_val
                            if not np.any(sub_col_mask):
                                if not already_a_full_row_constraint:
                                    construct_row_lambdas(sub_full_row_mask, sub_full_col_mask)
                                    already_a_full_row_constraint = True
                            else:
                                construct_row_lambdas(sub_full_row_mask, sub_col_mask)

                # Find lambda values using Newton optimization.
                P_sub = newton_optimization(lambdas, nit=100)

                P[np.ix_(row_mask, col_mask)] = P_sub
                P[np.ix_(col_mask, row_mask)] = P_sub.T

        self.__P = P

    def get_probability(self, row_ids, col_ids):
        return self.__P[row_ids, col_ids]

    def get_row_probability(self, row_ids, col_ids):
        return self.get_probability(row_ids, col_ids)

    def get_full_P_matrix(self):
        return self.__P

    def set_full_P_matrix(self, P):
        self.__P = P


def newton_optimization(lambdas, nit=100, tol=1e-8):
    alpha = 1.0
    prev_alpha = alpha
    P = None
    grad = None
    delta_la = None
    lagrangian = None
    start_time = time.time()
    for k in range(nit):
        # This is the first iteration, so calculate the initial values.
        if k == 0:
            E = lambdas.compute_E()
            lagrangian = lambdas.compute_lagrangian(E)
            P, grad, delta_la = lambdas.compute_P_and_grad(E)

        # Find the largest alpha that satisfies the first Wolfe condition.
        # This is done by halving alpha until it happens.
        while True:
            # Step in direction of gradient.
            lambdas.try_step(alpha)

            # Compute lagrangian with this alpha.
            E_try = lambdas.compute_E()
            lagrangian_try = lambdas.compute_lagrangian(E_try)

            # Check first Wolfe condition.
            if lagrangian_try <= lagrangian + 1e-4*alpha*(delta_la.dot(grad)):
                # print("lagrangian: "+str(lagrangian_try)+", alpha: "+str(alpha))

                # Condition is satisfied, the recently tried values are taken as the new current values.
                E = E_try
                lagrangian = lagrangian_try
                lambdas.finalize_step()
                P, grad, delta_la = lambdas.compute_P_and_grad(E)
                break
            else:
                alpha /= 2
                if alpha < 1e-8:
                    break

        # Some stop conditions.
        if np.linalg.norm(grad) / grad.shape[0] < tol or k >= nit - 1 or alpha < 1e-8:
            time_diff = time.time() - start_time
            print("Computed degree+block prior in " + str(k+1) +
                  " iterations (" + str(int(time_diff / 60)) + "m " + str(int(time_diff % 60)) + "s).")
            break

        # If the previous best alpha was the same as the current best alpha, then increase alpha.
        if prev_alpha == alpha:
            prev_alpha = alpha
            alpha = min(1.0, alpha*2)
        else:
            prev_alpha = alpha

    return P


class Lambdas:
    """
    General class for Lagrange multipliers or 'lambdas'.
    """
    def __init__(self):
        self.la = None
        self._delta_la = None
        self._backup_la = None

    def exponent_term(self):
        raise NotImplementedError

    def lagrangian_term(self):
        raise NotImplementedError

    def grad(self, partial_derivatives, P):
        raise NotImplementedError

    def try_step(self, alpha):
        if self._backup_la is None:
            self._backup_la = self.la
        self.la = self._backup_la + alpha * self._delta_la

    def finalize_step(self):
        self._delta_la = None
        self._backup_la = None


class RowDegreeLambdas(Lambdas):
    """
    For the constraint where the expected row sum (for the submatrix specified by row_mask and col_mask) is equal to the
    actual sum.
    """
    def __init__(self, degrees, row_mask, col_mask):
        super(RowDegreeLambdas, self).__init__()
        self.__degrees = degrees
        self.__row_mask = row_mask
        self.__col_mask = col_mask

        # Array of lambdas for row degrees.
        self.la = np.zeros(degrees.shape[0], dtype=np.float)

        # Initialization based on heuristics.
        P_estimate = (degrees + 1) / (np.sum(col_mask) + 1)
        self.la = np.log(P_estimate / (1 - P_estimate)) / 2

    def exponent_term(self):
        padded_la = np.zeros_like(self.__row_mask, dtype=np.float)
        padded_la[self.__row_mask] = self.la
        arranged_la = np.outer(padded_la, self.__col_mask)
        return arranged_la

    def lagrangian_term(self):
        return np.sum(self.la * self.__degrees)

    def grad(self, partial_derivatives, P):
        grad = (P.dot(self.__col_mask))[self.__row_mask] - self.__degrees
        hessian = (partial_derivatives.dot(self.__col_mask))[self.__row_mask]
        self._delta_la = -grad / (hessian + hessian.shape[0] * 1e-10)
        return self._delta_la, grad


class ColumnDegreeLambdas(Lambdas):
    """
    For the constraint where the expected column sum (for the submatrix specified by row_mask and col_mask) is equal to
    the actual sum.
    """
    def __init__(self, degrees, row_mask, col_mask):
        super(ColumnDegreeLambdas, self).__init__()
        self.__degrees = degrees
        self.__row_mask = row_mask
        self.__col_mask = col_mask

        # Array of lambdas for column degrees.
        self.la = np.zeros(degrees.shape[0], dtype=np.float)

        # Initialization based on heuristics.
        P_estimate = (degrees + 1) / (np.sum(row_mask) + 1)
        self.la = np.log(P_estimate / (1 - P_estimate)) / 2

    def exponent_term(self):
        padded_la = np.zeros_like(self.__col_mask, dtype=np.float)
        padded_la[self.__col_mask] = self.la
        return np.outer(self.__row_mask, padded_la)

    def lagrangian_term(self):
        return np.sum(self.la * self.__degrees)

    def grad(self, partial_derivatives, P):
        grad = (P.T.dot(self.__row_mask))[self.__col_mask] - self.__degrees
        hessian = (partial_derivatives.T.dot(self.__row_mask))[self.__col_mask]
        self._delta_la = -grad / (hessian + hessian.shape[0] * 1e-10)
        return self._delta_la, grad


class LambdasAggregator:
    """
    Perform aggregation operations on the lambdas objects. It is assumed that every element follows an independent
    Bernoulli distribution.
    """
    def __init__(self):
        self._lambdas_list = []
        self._backup_vals = None

    def add_lambdas_object(self, lambdas):
        assert isinstance(lambdas, Lambdas)
        self._lambdas_list.append(lambdas)

    def compute_E(self):
        exponent = 0
        for lambdas in self._lambdas_list:
            exp_term = lambdas.exponent_term()
            exponent += exp_term
        return np.exp(exponent)

    def compute_lagrangian(self, E):
        lag = np.log(self._Z(E))
        lag = np.sum(lag)
        for lambdas in self._lambdas_list:
            lag -= lambdas.lagrangian_term()
        return lag

    def compute_P_and_grad(self, E):
        Z = self._Z(E)
        P = E / Z
        partial_derivatives = E / (Z ** 2)

        delta_las = []
        grads = []
        for lambdas in self._lambdas_list:
            delta_la, grad = lambdas.grad(partial_derivatives, P)
            grads.append(grad)
            delta_las.append(delta_la)
        grad = np.concatenate(grads)
        delta_la = np.concatenate(delta_las)
        return P, grad, delta_la

    def try_step(self, alpha):
        for lambdas in self._lambdas_list:
            lambdas.try_step(alpha)

    def finalize_step(self):
        for lambdas in self._lambdas_list:
            lambdas.finalize_step()

    @staticmethod
    def _Z(E):
        """
        Calculate the partition function Z(lambda).
        """
        return 1 + E
