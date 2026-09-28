#!/usr/local/bin/python3
# -*- coding: utf-8 -*-
#
# archNEMESIS - Python implementation of the NEMESIS radiative transfer and retrieval code
# NestedSamplingTransport_0.py - Gaussian transport for posterior repartitioning.
#
# Copyright (C) 2026 Juan Alday, Joseph Penn, Patrick Irwin,
# Jack Dobinson, Jon Mason, Jingxuan Yang
#
# This file is part of archNEMESIS.
#
# archNEMESIS is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# You should have received a copy of the GNU General Public License
# along with this program. If not, see <https://www.gnu.org/licenses/>.

import numpy as np
import scipy.linalg
import scipy.special
import scipy.optimize


class NestedSamplingTransport_0:

    def coefficients(self):
        return dict(MEAN=self.MEAN.tolist(), CHOLESKY=self.CHOLESKY.tolist())


    @staticmethod
    def from_coefficients(coefficients):
        if 'WEIGHTS' in coefficients:
            return NestedSamplingMixture_0(coefficients['MEAN'], coefficients['CHOLESKY'], coefficients['WEIGHTS'])

        return NestedSamplingTransport_0(coefficients['MEAN'], coefficients['CHOLESKY'])


    def __init__(self, MEAN, CHOLESKY):
        """
        Gaussian auxiliary distribution in inverse-normal prior-cube coordinates.

        For independent Gaussian priors, z = (XN - XA) / XA_ERR
        is exactly Phi^-1(T^-1(XN)). Working directly in z avoids loss of
        precision from composing a normal CDF and its inverse in the tails.
        For other priors, NestedSamplingPrior_0 supplies the matching mappings
        between parameters and z = Phi^-1(T^-1(parameters)).
        """

        self.MEAN = np.array(MEAN, dtype=float, copy=True)
        self.CHOLESKY = np.array(CHOLESKY, dtype=float, copy=True)
        if self.MEAN.ndim != 1 or self.CHOLESKY.shape != (len(self.MEAN), len(self.MEAN)):
            raise ValueError('Transport mean and Cholesky factor have incompatible shapes')
        if not np.all(np.isfinite(self.MEAN)) or not np.all(np.isfinite(self.CHOLESKY)):
            raise ValueError('Transport coefficients must be finite')
        if not np.array_equal(self.CHOLESKY, np.tril(self.CHOLESKY)) or np.any(self.CHOLESKY.diagonal() <= 0.0):
            raise ValueError('Transport requires a lower triangular factor with positive diagonal')

        self.LOGDET = np.log(self.CHOLESKY.diagonal()).sum()


    @classmethod
    def from_samples(cls, samples, weights):
        """
        Fit to weighted guide samples in standardised prior coordinates.
        A small eigenvalue floor keeps the fitted density non-singular.
        """

        samples = np.asarray(samples, dtype=float)
        weights = np.asarray(weights, dtype=float)
        if samples.ndim != 2 or weights.shape != (len(samples),):
            raise ValueError('Expected a sample matrix and one posterior weight per row')
        if not np.all(np.isfinite(weights)) or np.any(weights < 0.0) or weights.sum() <= 0.0:
            raise ValueError('Guide posterior weights must be finite, non-negative and non-zero')

        keep = weights > 0.0
        samples = samples[keep]
        weights = weights[keep] / weights[keep].sum()
        if len(samples) < 2 or not np.all(np.isfinite(samples)):
            raise ValueError('Transport fit requires at least two finite guide samples')

        mean = np.sum(weights[:,None] * samples, axis=0)
        residual = samples - mean
        covariance = (residual * weights[:,None]).T @ residual
        eigenvalues, eigenvectors = np.linalg.eigh(covariance)
        floor = max(1.0e-12, eigenvalues.max() * 1.0e-8)
        covariance = (eigenvectors * np.maximum(eigenvalues, floor)) @ eigenvectors.T

        return cls(mean, np.linalg.cholesky(covariance))


    def Prior(self, cube):
        """
        Map the unit cube to auxiliary standardised-prior coordinates.
        """

        cube = np.asarray(cube, dtype=float)
        if cube.shape != self.MEAN.shape or np.any(~np.isfinite(cube)) or np.any((cube <= 0.0) | (cube >= 1.0)):
            raise ValueError('Transport input must lie in the open unit cube')

        return self.MEAN + self.CHOLESKY @ scipy.special.ndtri(cube)


    def InversePrior(self, z):
        """
        Map auxiliary standardised-prior coordinates back to the sampler cube.
        """

        v = scipy.linalg.solve_triangular(self.CHOLESKY, np.asarray(z) - self.MEAN, lower=True)

        return scipy.special.ndtr(v)


    def LogCorrection(self, z):
        """
        log pi_z(z) - log q_z(z), equal to log|det J_S|.
        The scientific-prior Jacobians cancel in this density ratio for any
        continuous invertible T used consistently to draw and evaluate samples.
        """

        z = np.asarray(z, dtype=float)
        v = scipy.linalg.solve_triangular(self.CHOLESKY, z - self.MEAN, lower=True)

        return self.LOGDET + 0.5 * (np.dot(v, v) - np.dot(z, z))


class NestedSamplingMixture_0:

    def __init__(self, MEAN, CHOLESKY, WEIGHTS):
        """
        Gaussian mixture in inverse-normal scientific-prior coordinates.
        Sequential conditional CDFs define a continuous, invertible cube map.
        """

        self.MEAN = np.array(MEAN, dtype=float, copy=True)
        self.CHOLESKY = np.array(CHOLESKY, dtype=float, copy=True)
        self.WEIGHTS = np.array(WEIGHTS, dtype=float, copy=True)
        if self.MEAN.ndim != 2:
            raise ValueError('Mixture means must have shape (components, parameters)')

        nk, ndim = self.MEAN.shape
        if nk < 1 or ndim < 1 or self.CHOLESKY.shape != (nk, ndim, ndim) or self.WEIGHTS.shape != (nk,):
            raise ValueError('Mixture coefficients have incompatible shapes')
        if not np.all(np.isfinite(self.WEIGHTS)) or np.any(self.WEIGHTS <= 0.0) or not np.isclose(self.WEIGHTS.sum(), 1.0):
            raise ValueError('Mixture weights must be positive and sum to one')

        for mean, factor in zip(self.MEAN, self.CHOLESKY):
            NestedSamplingTransport_0(mean, factor)

        self.WEIGHTS /= self.WEIGHTS.sum()
        self.LOGWEIGHTS = np.log(self.WEIGHTS)
        self.LOGDET = np.log(self.CHOLESKY.diagonal(axis1=1, axis2=2)).sum(axis=1)
        self.SELECTION = None


    def coefficients(self):
        return dict(MEAN=self.MEAN.tolist(), CHOLESKY=self.CHOLESKY.tolist(), WEIGHTS=self.WEIGHTS.tolist())


    def LogDensity(self, z):
        z = np.asarray(z, dtype=float)
        if z.ndim not in (1, 2) or z.shape[-1] != self.MEAN.shape[1] or not np.all(np.isfinite(z)):
            raise ValueError('Invalid coordinates for mixture density')

        points = np.atleast_2d(z)
        values = np.empty((len(points), len(self.WEIGHTS)))

        for k in range(len(self.WEIGHTS)):
            white = scipy.linalg.solve_triangular(self.CHOLESKY[k], (points-self.MEAN[k]).T, lower=True).T
            values[:,k] = self.LOGWEIGHTS[k] - self.LOGDET[k] - 0.5*(np.sum(white**2, axis=1) + z.shape[-1]*np.log(2.0*np.pi))

        result = scipy.special.logsumexp(values, axis=1)

        return float(result[0]) if z.ndim == 1 else result


    def LogCorrection(self, z):
        z = np.asarray(z, dtype=float)

        return -0.5*(np.dot(z, z) + len(z)*np.log(2.0*np.pi)) - self.LogDensity(z)


    def _map(self, values, inverse):
        values = np.asarray(values, dtype=float)
        ndim = self.MEAN.shape[1]
        if values.shape != (ndim,) or not np.all(np.isfinite(values)):
            raise ValueError('Invalid coordinates for mixture transform')
        if not inverse and np.any((values <= 0.0) | (values >= 1.0)):
            raise ValueError('Transport input must lie in the open unit cube')

        result = np.empty(ndim)
        white = np.zeros_like(self.MEAN)
        logweights = self.LOGWEIGHTS.copy()

        for j in range(ndim):
            mean = self.MEAN[:,j] + np.sum(self.CHOLESKY[:,j,:j]*white[:,:j], axis=1)
            sigma = self.CHOLESKY[:,j,j]

            if inverse:
                value = values[j]
                residual = (value-mean)/sigma
                logcdf = scipy.special.logsumexp(logweights + scipy.special.log_ndtr(residual))
                logsf = scipy.special.logsumexp(logweights + scipy.special.log_ndtr(-residual))
                result[j] = np.exp(logcdf) if logcdf < np.log(0.5) else -np.expm1(logsf)
            else:
                u = values[j]
                quantiles = mean + sigma*scipy.special.ndtri(u)
                lower, upper = quantiles.min(), quantiles.max()

                if lower == upper:
                    value = lower
                else:
                    # Log-CDF and log-survival calculations retain tail precision.
                    if u <= 0.5:
                        def objective(value):
                            return scipy.special.logsumexp(logweights + scipy.special.log_ndtr((value-mean)/sigma)) - np.log(u)
                    else:
                        def objective(value):
                            return np.log1p(-u) - scipy.special.logsumexp(logweights + scipy.special.log_ndtr((mean-value)/sigma))

                    # Slight padding handles roundoff when one component dominates.
                    padding = 1.0e-12*max(1.0, abs(lower), abs(upper))
                    value = scipy.optimize.brentq(objective, lower-padding, upper+padding, xtol=1.0e-12)

                result[j] = value
                residual = (value-mean)/sigma

            white[:,j] = residual
            logweights += -0.5*residual**2 - np.log(sigma)
            logweights -= scipy.special.logsumexp(logweights)

        return result


    def Prior(self, cube):
        return self._map(cube, inverse=False)


    def InversePrior(self, z):
        return self._map(z, inverse=True)


    @classmethod
    def _fit(cls, samples, weights, n_components, seed, n_init=3):
        """
        Weighted EM in globally whitened coordinates; regularise each component.
        The original posterior weights enter every sufficient statistic.
        """

        weights = weights/weights.sum()
        global_fit = NestedSamplingTransport_0.from_samples(samples, weights)
        if n_components == 1:
            return cls(global_fit.MEAN[None,:], global_fit.CHOLESKY[None,:,:], [1.0])

        x = scipy.linalg.solve_triangular(global_fit.CHOLESKY, (samples-global_fit.MEAN).T, lower=True).T
        ndim = x.shape[1]
        rng = np.random.default_rng(seed)
        best, best_score = None, -np.inf

        for start in range(n_init):
            means = [x[rng.choice(len(x), p=weights)]]

            for k in range(1, n_components):
                distance = np.min(np.sum((x[:,None,:]-np.array(means)[None,:,:])**2, axis=2), axis=1)
                probability = weights*distance
                if probability.sum() <= 0.0:
                    raise ValueError('Too few distinct guide samples for mixture')

                means.append(x[rng.choice(len(x), p=probability/probability.sum())])

            means = np.array(means)
            covariance = np.tile(np.eye(ndim), (n_components, 1, 1))
            mixing = np.full(n_components, 1.0/n_components)
            previous = -np.inf

            for iteration in range(400):
                factors = np.linalg.cholesky(covariance)
                logprob = np.empty((len(x), n_components))

                for k in range(n_components):
                    white = scipy.linalg.solve_triangular(factors[k], (x-means[k]).T, lower=True).T
                    logprob[:,k] = (
                        np.log(mixing[k]) - np.log(factors[k].diagonal()).sum()
                        - 0.5*(np.sum(white**2, axis=1) + ndim*np.log(2.0*np.pi))
                    )

                logdensity = scipy.special.logsumexp(logprob, axis=1)
                score = float(weights @ logdensity)
                if abs(score-previous) < 1.0e-6:
                    break

                previous = score
                responsibility = np.exp(logprob-logdensity[:,None])*weights[:,None]
                mixing = responsibility.sum(axis=0)
                if mixing.min() < 1.0e-10:
                    break

                means = (responsibility.T @ x)/mixing[:,None]

                for k in range(n_components):
                    residual = x-means[k]
                    covariance[k] = (residual*responsibility[:,k,None]).T @ residual/mixing[k]
                    eigenvalues, eigenvectors = np.linalg.eigh(covariance[k])
                    covariance[k] = (eigenvectors*np.maximum(eigenvalues, 1.0e-6)) @ eigenvectors.T
            else:
                continue

            if mixing.min() < 1.0e-10 or not np.isfinite(score):
                continue

            candidate = cls(global_fit.MEAN + means @ global_fit.CHOLESKY.T,
                            np.array([global_fit.CHOLESKY @ factor for factor in np.linalg.cholesky(covariance)]), mixing)
            score = float(weights @ candidate.LogDensity(samples))

            if score > best_score:
                best, best_score = candidate, score

        if best is None:
            raise ValueError('Weighted Gaussian mixture fitting did not converge')

        return best


    @classmethod
    def from_samples(cls, samples, weights, max_components=6, seed=0):
        """
        Select component count by five-fold weighted predictive log density.
        Prefer the smallest model within one fold standard error of the best.
        Fold variation is a selection heuristic, not an independent-sample error.
        """

        samples = np.asarray(samples, dtype=float)
        weights = np.asarray(weights, dtype=float)
        NestedSamplingTransport_0.from_samples(samples, weights)
        keep = weights > 0.0

        # Keep duplicate coordinates together so they cannot leak across folds.
        samples, inverse = np.unique(samples[keep], axis=0, return_inverse=True)
        weights = np.bincount(inverse, weights=weights[keep])
        weights /= weights.sum()
        if not isinstance(max_components, (int, np.integer)) or isinstance(max_components, (bool, np.bool_)) or max_components < 1:
            raise ValueError('max_components must be a positive integer')

        neff = float(1.0/np.sum(weights**2))
        if len(samples) < 10 or neff < 20.0:
            raise ValueError('Too few effective guide samples for mixture selection')

        rng = np.random.default_rng(seed)
        folds = np.empty(len(samples), dtype=int)
        folds[rng.permutation(len(samples))] = np.arange(len(samples)) % 5
        scores = []
        ndim = samples.shape[1]

        for nk in range(1, max_components+1):
            n_parameters = nk*(ndim + ndim*(ndim+1)//2 + 1)-1
            if nk > 1 and any((weights[folds != f].sum()**2/np.sum(weights[folds != f]**2)) < 2*n_parameters for f in range(5)):
                break

            values = []

            try:
                for fold in range(5):
                    train = folds != fold
                    fit = cls._fit(samples[train], weights[train], nk, seed+100*nk+fold)
                    values.append(float(np.average(fit.LogDensity(samples[~train]), weights=weights[~train])))
            except ValueError as exception:
                scores.append(dict(components=nk, error=str(exception)))
                continue

            scores.append(dict(
                components=nk,
                mean=float(np.mean(values)),
                standard_error=float(np.std(values, ddof=1)/np.sqrt(5)),
                folds=values
            ))

        valid = [row for row in scores if 'mean' in row]
        if not valid:
            raise ValueError('No mixture candidate passed cross-validation')

        best = max(valid, key=lambda row: row['mean'])
        selected = min(row['components'] for row in valid if row['mean'] >= best['mean']-best['standard_error'])
        fit = cls._fit(samples, weights, selected, seed+10000, n_init=5)
        fit.SELECTION = dict(
            method='five_fold_weighted_log_density_one_standard_error',
            seed=seed,
            effective_samples=neff,
            max_components=max_components,
            selected_components=selected,
            scores=scores
        )

        return fit
