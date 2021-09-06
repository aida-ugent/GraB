import numpy as np

from embedding.cne_recommender.bg_dist_partite import BgDistBlockAbstract, newton_optimization


class BgDistDegreeBlockEco(BgDistBlockAbstract):
    # Constraint(s): total block sum and total node degree.

    def __init__(self, A, block_mask):
        super(BgDistDegreeBlockEco, self).__init__(A, block_mask)

        self.__P = None

    def fit(self):
        block_types = np.unique(self._block_mask)
        assert np.all(block_types == np.arange(block_types.shape[0]))

        P = np.empty(self._A.shape, dtype=np.float)
        P[:] = np.nan

        for type_i in range(block_types.shape[0]):
            row_mask = self._block_mask == type_i
            for type_j in range(type_i, block_types.shape[0]):
                col_mask = self._block_mask == type_j

                # First, check if the total sum is not just zero. In that case, we can avoid doing any calculations.
                total_sum = row_mask.dot(self._A.dot(col_mask))
                if total_sum == 0:
                    P[np.ix_(row_mask, col_mask)] = 0.0
                    P[np.ix_(col_mask, row_mask)] = 0.0
                    continue

                # Compute the row sum for the given submatrix. The expected row sum will have to match the actual sum.
                # Note that this is done by doing a (matrix x column_vector) multiplication, which sums all rows in A
                # for the specified col_mask indices. Of these, only the rows in row_mask are kept.
                row_sums = (self._A.dot(col_mask))[row_mask]

                # Construct Lambdas object for the row degree prior.
                row_lambdas = RowDegreeLambdas(row_sums)

                # The col_sums are computed in a similar way.
                col_sums = (self._A.T.dot(row_mask))[col_mask]

                # Construct Lambdas object for the column degree prior.
                col_lambdas = ColumnDegreeLambdas(col_sums)

                # Lamdas aggregator keeps track of several 'lambdas' objects.
                lambdas = LambdasAggregator()

                lambdas.add_lambdas_object(row_lambdas)
                lambdas.add_lambdas_object(col_lambdas)
                lambdas.compile()

                # Find lambda values using Newton optimization.
                P_sub_eco = newton_optimization(lambdas, nit=500)
                P_sub = P_sub_eco[np.ix_(row_lambdas.idx_to_uni, col_lambdas.idx_to_uni)]

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


class Lambdas:
    """
    General class for Lagrange multipliers or 'lambdas'.
    """
    def __init__(self, constraints):
        # Find unique constraints (cs).
        uni_cs, idx_to_uni, cs_counts = np.unique(constraints, return_inverse=True, return_counts=True)
        self._uni_cs = uni_cs
        self.idx_to_uni = idx_to_uni
        self.multip = cs_counts

        self.la = np.zeros(uni_cs.shape[0], dtype=np.float)
        self._delta_la = None
        self.__backup_la = None

        self._row_multip = None
        self._col_multip = None

    def set_counts(self, row_counts, col_counts):
        self._row_multip = row_counts
        self._col_multip = col_counts

    def exponent_term(self):
        raise NotImplementedError

    def lagrangian_term(self):
        raise NotImplementedError

    def grad(self, partial_derivatives, P):
        raise NotImplementedError

    def try_step(self, alpha):
        if self.__backup_la is None:
            self.__backup_la = self.la
        self.la = self.__backup_la + alpha * self._delta_la

    def finalize_step(self):
        self._delta_la = None
        self.__backup_la = None


# class DegreeLambdas(Lambdas):
#     """
#     For the constraint where the expected total degree of each node is equal to the actual total degree.
#     """
#     def __init__(self, degrees):
#         super(DegreeLambdas, self).__init__(degrees)
#
#     def exponent_term(self):
#         return self.la[:, np.newaxis] / 2 + self.la[np.newaxis, :] / 2
#
#     def lagrangian_term(self):
#         return np.sum(self.la * self._uni_cs * self._cs_counts)
#
#     def grad(self, partial_derivatives, P):
#         grad = (P.dot(self._cs_counts) - self._uni_cs) #* self._cs_counts
#         hessian = (1 / 2) * partial_derivatives.dot(self._cs_counts)
#         self._delta_la = -grad / (hessian + np.ones_like(hessian) * hessian.shape[0] * 1e-10)
#         return self._delta_la, grad


class RowDegreeLambdas(Lambdas):
    """
    For the constraint where the expected row sum (for the submatrix specified by row_mask and col_mask) is equal to the
    actual sum.
    """
    def __init__(self, degrees):
        super(RowDegreeLambdas, self).__init__(degrees)

    def exponent_term(self):
        return self.la[:, np.newaxis]

    def lagrangian_term(self):
        return np.sum(self.la * self._uni_cs * self._row_multip)

    def grad(self, partial_derivatives, P):
        grad = P.dot(self._col_multip) - self._uni_cs
        hessian = (partial_derivatives.dot(self._col_multip))
        self._delta_la = -grad / (hessian + hessian.shape[0] * 1e-10)
        return self._delta_la, grad


class ColumnDegreeLambdas(Lambdas):
    """
    For the constraint where the expected column sum (for the submatrix specified by row_mask and col_mask) is equal to
    the actual sum.
    """
    def __init__(self, degrees):
        super(ColumnDegreeLambdas, self).__init__(degrees)

    def exponent_term(self):
        return self.la[np.newaxis, :]

    def lagrangian_term(self):
        return np.sum(self.la * self._uni_cs * self._col_multip)

    def grad(self, partial_derivatives, P):
        grad = P.T.dot(self._row_multip) - self._uni_cs
        hessian = (partial_derivatives.T.dot(self._row_multip))
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

        self._row_multip = None
        self._col_multip = None

    def add_lambdas_object(self, lambdas):
        assert isinstance(lambdas, Lambdas)
        self._lambdas_list.append(lambdas)

    def compile(self):
        self._row_multip = self._lambdas_list[0].multip
        self._col_multip = self._lambdas_list[1].multip
        for lambdas in self._lambdas_list:
            lambdas.set_counts(self._row_multip, self._col_multip)

    def compute_E(self):
        exponent = 0
        for lambdas in self._lambdas_list:
            exp_term = lambdas.exponent_term()
            exponent = exponent + exp_term
        return np.exp(exponent)

    def compute_lagrangian(self, E):
        lag = self._row_multip.dot(np.log(self._Z(E))).dot(self._col_multip)
        for lambdas in self._lambdas_list:
            lag -= lambdas.lagrangian_term()
        return lag

    def compute_P_and_grad(self, E):
        Z = self._Z(E)
        P = E / Z

        partial_derivatives = P / Z

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
