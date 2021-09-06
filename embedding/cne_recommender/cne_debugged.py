import os
import numpy as np
from collections import defaultdict
from functools import partial
from multiprocessing import Pool


import numpy as np

from .bg_dist_partite import BgDistBuilder
from .utils import compute_tr_val_split, validate
from ..cne.degree_prior import BackgroundDistribution

class ConditionalNetworkEmbedding:
    def __init__(self, A, d, s1, s2, prior_dist):
        self.__A = A
        self.__d = d
        self.__s1 = s1
        self.__s2 = s2
        self.__prior_dist = prior_dist

    def _subsample(self, A, neg_pos_ratio):
        sids_list = []
        pos_ratios = []
        neg_ratios = []

        n = A.shape[0]
        for aid in range(n):
            samples = []
            nbr_ids = A.indices[A.indptr[aid]:A.indptr[aid+1]]
            num_pos = len(nbr_ids)
            samples.extend(nbr_ids)

            neg_samples = list(set(np.random.randint(n, size=(num_pos*neg_pos_ratio,))) - set(nbr_ids) - set([aid]))
            samples.extend(neg_samples)
            sids_list.append(samples)

            pos_ratios.append(len(nbr_ids)/len(nbr_ids))
            neg_ratios.append((n - len(nbr_ids) - 1)/len(neg_samples))

        return sids_list, pos_ratios, neg_ratios

    def _obj_grad(self, X, A, prior_dist, s1, s2, sids_list=None,
                  pos_ratios=None, neg_ratios=None, reweigh=True):
            res_obj = 0.
            res_grad = np.zeros_like(X)
            n = X.shape[0]
            for xid in range(n):
                sids = range(n) if sids_list is None else sids_list[xid]
                prior = prior_dist.get_row_probability(xid, sids)

                diff = (X[xid, :] - X[sids, :]).T
                d_p2 = np.sum(diff**2, axis=0)

                post = self._posterior(0, d_p2, prior, s1, s2)
                nbr_ids = self._nbr_ids(A, xid, sids)
                post[nbr_ids] = 1-post[nbr_ids]
                obj = np.log(post+1e-20)
                if sids_list is not None and reweigh:
                    obj *= neg_ratios[xid]
                    obj[nbr_ids] *= pos_ratios[xid]/neg_ratios[xid]
                res_obj += np.sum(obj)

                grad_coeff = 1 - post
                grad_coeff[nbr_ids] *= -1
                grad = (1/s1**2 - 1/s2**2)*(grad_coeff*diff).T
                if sids_list is not None and reweigh:
                    grad *= neg_ratios[xid]
                    grad[nbr_ids] *= pos_ratios[xid]/neg_ratios[xid]
                res_grad[xid, :] += np.sum(grad, axis=0)
                res_grad[sids, :] -= grad
            return -res_obj, -res_grad

    def _obj_grad_optimized(self, X, prior_dist, s1, s2, total_n_ids):
        # no sampling
        res_obj = 0.
        n = X.shape[0]
        total_diff = (np.repeat(X, repeats=X.shape[0], axis=0) - np.tile(X, (X.shape[0], 1))).T
        d_p2_total = np.sum(total_diff ** 2, axis=0)
        prior_total = prior_dist.get_full_P_matrix().flatten()
        post_total = self._posterior(0, d_p2_total, prior_total, s1, s2)

        post_total[total_n_ids] = 1-post_total[total_n_ids]

        obj = np.log(post_total+1e-20)
        res_obj += np.sum(obj)

        grad_coeff = 1 - post_total
        grad_coeff[total_n_ids] *= -1
        grad = (1/s1**2 - 1/s2**2)*(grad_coeff*total_diff).T

        res_grad = np.add.reduceat(grad, np.arange(0, grad.shape[0], n), axis=0)
        res_grad -= np.sum(grad.reshape((n, n, X.shape[1])), axis=(0))

        # for xid in range(n):
        # res_grad[xid, :] += np.sum(grad[xid*n:(xid+1)*n, :], axis=0)
        # res_grad -= grad[xid*n:(xid+1)*n, :]
        return -res_obj, -res_grad

    def _row_posterior(self, row_id, col_ids, X, prior_dist, s1, s2):
        prior = prior_dist.get_row_probability(row_id, col_ids)
        d_p2 = np.sum(((X[row_id, :] - X[col_ids, :]).T)**2, axis=0)
        return self._posterior(1, d_p2, prior, s1, s2)

    def _posterior(self, obs_val, d_p2, prior, s1, s2):
        s_div = s1/s2
        s_diff = (1/s1**2 - 1/s2**2)
        if obs_val == 1:
            return 1./(1+(1-prior)/prior*s_div*np.exp(d_p2/2*s_diff))
        else:
            return 1./(1+prior/(1-prior)/s_div*np.exp(-d_p2/2*s_diff))

    def _nbr_ids(self, csr_A, aid, sids):
        nbr_ids = csr_A.indices[csr_A.indptr[aid]:csr_A.indptr[aid+1]]
        if len(sids) != csr_A.shape[0]:
            nbr_ids = np.where(np.in1d(sids, nbr_ids, assume_unique=True))[0]
        return nbr_ids

    def _optimizer_adam(self, X, A, prior_dist, s1, s2, num_epochs=2000,
                        alpha=0.2, beta_1=0.9, beta_2=0.9999, eps=1e-8,
                        ftol=1e-3, w=10, gamma=10, subsample=True,
                        neg_pos_ratio=5, verbose=True):
        m_prev = np.zeros_like(X)
        v_prev = np.zeros_like(X)
        obj_old = 0.

        total_n_ids = []
        ids = []
        n = X.shape[0]
        for xid in range(n):
            nbr_ids = A.indices[A.indptr[xid]:A.indptr[xid+1]]
            total_n_ids += list((xid*n + nbr_ids))
            ids += [xid for _ in range(n)]

        for epoch in range(num_epochs):
            obj, grad = self._obj_grad_optimized(X, prior_dist, s1, s2, total_n_ids)

            # Adam optimizer
            m = beta_1*m_prev + (1-beta_1)*grad
            v = beta_2*v_prev + (1-beta_2)*grad**2

            m_prev = m.copy()
            v_prev = v.copy()

            m = m/(1-beta_1**(epoch+1))
            v = v/(1-beta_2**(epoch+1))
            X -= alpha*m/(v**.5 + eps)

            grad_norm = np.sum(grad**2)**.5

            obj_smooth = np.abs(obj_old - obj)
            obj_old = obj
            if verbose:
                print('Epoch: {:d}, grad norm: {:.4f}, obj: {:.4f}, obj smoothness: {:.4f}'.format(epoch, grad_norm, obj, obj_smooth), flush=True)
            if obj_smooth < ftol:
                break
        return X

    def tune(self, s2s, portion=0.9, n_workers=None):
        tr_A, tr_E, val_E = compute_tr_val_split(self.__A, portion)
        n_workers = os.cpu_count()-2 if n_workers is None else n_workers
        with Pool(n_workers) as p:
            res = p.map(partial(validate, self._optimizer_adam, self._predict,
                                tr_A, tr_E, val_E, self.__prior_dist, self.__d, self.__s1), s2s)
        self.__s2 = s2s[np.argmax(res)]
        return self.__s2

    def fit(self, lr=0.1, max_iter=2000, ftol=1e-3, subsample=False,
            neg_pos_ratio=5, verbose=True, init_embeddings=None, ):
        if init_embeddings is None:
            X0 = np.random.randn(self.__A.shape[0], self.__d)
        else:
            X0 = init_embeddings
        self.__emb = self._optimizer_adam(X0, self.__A, self.__prior_dist,
            self.__s1, self.__s2, alpha=lr, num_epochs=max_iter,
            ftol=ftol, subsample=subsample, neg_pos_ratio=neg_pos_ratio,
            verbose=verbose)

    def predict(self, E):
        return self._predict(E, self.__emb, self.__prior_dist, self.__s1,
                             self.__s2)

    def _predict(self, E, emb, prior_dist, s1, s2):
        edge_dict = defaultdict(list)
        ids_dict = defaultdict(list)
        for i, edge in enumerate(E):
            edge_dict[edge[0]].append(edge[1])
            ids_dict[edge[0]].append(i)

        pred = []
        ids = []
        for u in edge_dict.keys():
            pred.extend(self._row_posterior(u, edge_dict[u], emb, prior_dist, s1, s2))
            ids.extend(ids_dict[u])

        return [p for _,p in sorted(zip(ids, pred))]

    @property
    def sids_list(self):
        return self.__sids_list

    @property
    def pos_ratios(self):
        return self.__pos_ratios

    @property
    def neg_ratios(self):
        return self.__neg_ratios

    def get_embeddings(self):
        return self.__emb

    def set_adj_matrix(self, A):
        self.__A = A

    def get_adj_matrix(self):
        return self.__A

    def get_posterior_row_col(self, row_id, col_id):
        prior = self.__prior_dist.get_row_probability(row_id, [col_id])
        d_p2 = np.sum(((self.__emb[row_id, :] - self.__emb[col_id, :]).T)**2, axis=0)
        return self._posterior(1, d_p2, prior, self.__s1, self.__s2)[0]

    def get_posterior_row(self, row_id, prior='default'):
        """

        :param row_id:
        :param prior: default, block, uniform
        :return:
        """
        if prior == 'uniform':
            if not hasattr(self, '__uniform_prior_dist'):
                self._set_uniform_prior()
            return self._row_posterior(row_id, range(self.__A.shape[0]), self.__emb, self.__uniform_prior_dist, self.__s1, self.__s2)
        elif prior == 'block':
            if not hasattr(self, '__block_prior_dist'):
                self._set_block_prior()
            return self._row_posterior(row_id, range(self.__A.shape[0]), self.__emb, self.__block_prior_dist, self.__s1, self.__s2)
        else:
            return self._row_posterior(row_id, range(self.__A.shape[0]), self.__emb, self.__prior_dist, self.__s1, self.__s2)

    def _set_uniform_prior(self):
        self.__uniform_prior_dist = BackgroundDistribution(self.__A, 'uniform')
        self.__uniform_prior_dist.fit()

    def _set_block_prior(self):
        self.__block_prior_dist = BgDistBuilder.build(self.__A, 'block', block_mask=self.__prior_dist._block_mask)
        self.__block_prior_dist.fit()
