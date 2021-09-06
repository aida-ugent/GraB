from __future__ import division, print_function, absolute_import

import os
import random
import subprocess

from scipy.stats import gaussian_kde

from scipy import linalg, special
from scipy._lib._numpy_compat import cov

from numpy import (asarray, atleast_2d, reshape, zeros, newaxis, dot, exp, pi,
                   sqrt, ravel, power, atleast_1d, squeeze, sum, transpose,
                   ones)
import numpy as np


class gaussian_kde_custom(gaussian_kde):
    def set_bandwidth(self, bw_method=None):
        self.bw_method = bw_method
        if not self.bw_method:
            self.bw_method = 'scott'
        if self.bw_method == '':
            self.bw_method = 'scott'
        super(gaussian_kde_custom, self).set_bandwidth(bw_method=self.bw_method)


    def _compute_covariance(self):
        """Computes the covariance matrix for each Gaussian kernel using
        covariance_factor().
        """
        self.factor = self.covariance_factor()
        # Cache covariance and inverse covariance of the data
        if not hasattr(self, '_data_inv_cov'):
            self._data_covariance = atleast_2d(cov(self.dataset, rowvar=1,
                                               bias=False,
                                               aweights=self.weights))
            self._data_covariance = self._data_covariance * np.identity(self._data_covariance.shape[0])
            self._data_inv_cov = linalg.inv(self._data_covariance)

        self.covariance = self._data_covariance * self.factor**2  # H
        self.inv_cov = self._data_inv_cov / self.factor**2
        self._norm_factor = sqrt(linalg.det(power(2*pi, self.d)*self.covariance))

    def evaluate(self, points):
        """Evaluate the estimated pdf on a set of points.

        Parameters
        ----------
        points : (# of dimensions, # of points)-array
            Alternatively, a (# of dimensions,) vector can be passed in and
            treated as a single point.

        Returns
        -------
        values : (# of points,)-array
            The values at each point.

        Raises
        ------
        ValueError : if the dimensionality of the input points is different than
                     the dimensionality of the KDE.

        """
        points = atleast_2d(asarray(points))

        d, m = points.shape
        if d != self.d:
            if d == 1 and m == self.d:
                # points was passed in as a row vector
                points = reshape(points, (self.d, 1))
                m = 1
            else:
                msg = "points have dimension %s, dataset has dimension %s" % (d,
                    self.d)
                raise ValueError(msg)

        result = zeros((m,), dtype=float)

        # whitening = linalg.cholesky(self.inv_cov)
        # scaled_dataset = dot(whitening, self.dataset)
        # scaled_points = dot(whitening, points)

        # loop over points
        for i in range(m):
            diff = points[:, i, newaxis] - self.dataset
            energy = np.sum(np.multiply(np.matmul(diff.T, self.inv_cov), diff.T), axis=1) / 2.0
            # energy = sum(diff * diff, axis=0) / 2.0
            result[i] = sum(exp(-energy)*self.weights, axis=0)

        result = result / self._norm_factor

        return result

    __call__ = evaluate

    def gradient(self, points):
        """Evaluate the estimated pdf gradient on a set of points.

        Parameters
        ----------
        points : (# of dimensions, # of points)-array
            Alternatively, a (# of dimensions,) vector can be passed in and
            treated as a single point.

        Returns
        -------
        values : (# of points,)-array
            The values at each point.

        Raises
        ------
        ValueError : if the dimensionality of the input points is different than
                     the dimensionality of the KDE.

        """
        points = atleast_2d(asarray(points))

        d, m = points.shape
        if d != self.d:
            if d == 1 and m == self.d:
                # points was passed in as a row vector
                points = reshape(points, (self.d, 1))
                m = 1
            else:
                msg = "points have dimension %s, dataset has dimension %s" % (d,
                    self.d)
                raise ValueError(msg)

        result = zeros((m,self.d), dtype=float)

        # whitening = linalg.cholesky(self.inv_cov)
        # scaled_dataset = dot(whitening, self.dataset)
        # scaled_points = dot(whitening, points)

        # loop over points
        for i in range(m):
            diff = points[:, i, newaxis] - self.dataset
            energy = np.sum(np.multiply(np.matmul(diff.T, self.inv_cov), diff.T), axis=1)
            result[i] = sum(np.matmul(diff.T, (self.inv_cov+self.inv_cov.T))*np.tile(exp(-energy),(self.d, 1)).T*np.tile(self.weights,(self.d, 1)).T, axis=0)

        result = result / (-2*self.neff*self._norm_factor)

        return result