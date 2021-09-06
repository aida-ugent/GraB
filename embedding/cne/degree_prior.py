from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import logging

import numpy as np
from scipy import sparse

logger = logging.getLogger(__name__)


class BackgroundDistribution:
    def __init__(self, A, prior, mask=None):
        '''
        input
        -----
        A: adjacency matrix, preferably CSR (compressed sparse row) matrix.
        prior: prior type
        '''
        self.__A = sparse.csr_matrix(A)
        self.__A.setdiag(0)
        self.__A.eliminate_zeros()
        self.__A.data = np.abs(self.__A.data)     # Make all edges positive
        self.__prior = prior
        self.__mask = mask

    def fit(self):
        if self.__prior == 'degree':
            self.fit_degree()
        elif self.__prior == 'uniform':
            self.fit_uniform()
        else:
            raise ValueError('Prior {:s} is not implemend'.format(self.__prior))

    def fit_uniform(self):
        n = self.__A.shape[1]
        if self.__mask is not None:
            self.__uniform_prob = np.sum(np.abs(self.__A.data)) / self.__mask.nnz
        else:
            self.__uniform_prob = np.sum(np.abs(self.__A.data)) / (n ** 2 - n)

    def fit_degree(self, nit=100, tol=1e-5):
        logger.info('Fit background distribution with degree prior...')
        n = self.__A.shape[1]
        # compute margin
        prows = np.array(self.__A.sum(axis=1).T.tolist()[0])/n

        # compute equivalent class
        [prowsunique, irows, jrows] = np.unique(prows, return_index=True,
                                                return_inverse=True)
        self.__la_inverse_ids = jrows

        nunique = len(prowsunique)
        la = np.zeros((1, nunique))

        vrows = []
        for i in range(nunique):
            vrows.append(len(np.where(jrows == i)[0]))

        lb = -5
        errors = []
        for k in range(nit-1):
            # compute gradient
            E = np.tile(np.exp(la/2.), (nunique, 1)) * \
                np.tile(np.exp(la/2.), (nunique, 1)).T
            ps = E/(1+E)
            gla = (-n*prowsunique+np.dot(ps, vrows)-np.diag(ps))*vrows
            errors.append(np.linalg.norm(gla))

            # compute Hessian
            H = 1./2*np.dot(np.dot(np.diag(vrows), E/(1+E)**2), np.diag(vrows))
            H = H+np.diag(np.sum(H, axis=1))-2*np.diag(np.diag(H)/vrows)
            H = H+np.trace(H)/nunique*1e-10

            # update gradient
            deltala = -np.dot(np.linalg.inv(H), gla)

            # compute the learning rate
            fbest = 0
            errorbest = errors[k]
            for f in np.logspace(lb, 1, 20):
                latry = la + f*deltala
                Etry = np.tile(np.exp(latry/2), (nunique, 1)) * \
                    np.tile(np.exp(latry/2), (nunique, 1)).T
                pstry = Etry/(1+Etry)
                glatry = (-n*prowsunique+np.dot(pstry, vrows) -
                          np.diag(pstry))*vrows
                errortry = np.linalg.norm(glatry)
                if errortry < errorbest:
                    fbest = f
                    errorbest = errortry
            if fbest == 0:
                lb *= 2
            la += fbest * deltala

            if errors[k]/n < tol:
                break

        self.__opt_la = la[0]

    def __get_prob_row_degree(self, row_id, col_ids):
        '''
        Compute prior (degree) probability for the entries in a row specified
        by row_id.
        '''
        row_la = self.__opt_la[self.__la_inverse_ids[row_id]]
        col_las = self.__opt_la[self.__la_inverse_ids[col_ids]]
        E = np.exp(row_la/2 + col_las/2)
        P = E/(1+E)
        return P

    def __get_prob_row_uniform(self, row_id, col_ids):
        return np.ones_like(col_ids) * self.__uniform_prob

    def get_row_probability(self, row_id, col_ids):
        '''
        Compute prior probability for the entries in a row specified by row_id.
        We assume no diagonal entry is indexed, i.e., row_id is not in col_ids.
        '''
        if self.__prior == 'degree':
            return self.__get_prob_row_degree(row_id, col_ids)
        elif self.__prior == 'uniform':
            return self.__get_prob_row_uniform(row_id, col_ids)
        else:
            raise ValueError('Prior {:s} is not implemented'
                             .format(self.__prior))

    def get_probability(self, row_id, col_ids):
        return self.get_row_probability(row_id, col_ids)

