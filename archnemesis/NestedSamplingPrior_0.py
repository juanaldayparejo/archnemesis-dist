#!/usr/local/bin/python3
# -*- coding: utf-8 -*-
#
# archNEMESIS - Python implementation of the NEMESIS radiative transfer and retrieval code
# NestedSamplingPrior_0.py - Scientific priors for nested sampling.
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

import os
import numpy as np
import scipy.special
import scipy.stats


class NestedSamplingPrior_0:

    def __init__(self, Variables):
        """
        Independent scientific priors, separate from the .apr/OE arrays.
        Unspecified entries retain the legacy Gaussian prior and error threshold.
        """

        self.XN = np.array(Variables.XN, dtype=float, copy=True)
        self.LX = np.array(Variables.LX, dtype=int, copy=True)
        self.NXVAR = np.array(Variables.NXVAR, dtype=int, copy=True)
        self.HAS_FILE = False
        self.DEFINITIONS = []
        self.DISTRIBUTIONS = []

        errors = np.sqrt(np.diag(Variables.SA))
        if not np.all(np.isfinite(Variables.XA)) or not np.all(np.isfinite(errors)) or not np.all(np.isfinite(self.XN)):
            raise ValueError('Prior values and standard deviations must be finite')
        if np.any(self.NXVAR < 0) or self.NXVAR.sum() != len(self.XN):
            raise ValueError('NXVAR must describe every state-vector entry')

        ix = 0

        for ivar,nx in enumerate(self.NXVAR):
            for element in range(nx):
                if errors[ix] > 1.0e-5:
                    name = 'GAUSSIAN'
                    arguments = [float(Variables.XA[ix]), float(errors[ix])]
                else:
                    name = 'FIXED'
                    arguments = [float(self.XN[ix])]

                self.DEFINITIONS.append(dict(variable=ivar+1, element=element+1, index=ix,
                                             coordinates='STATE', distribution=name,
                                             arguments=arguments, LX=int(self.LX[ix])))
                self.DISTRIBUTIONS.append(self.make_distribution('STATE', name, arguments, self.LX[ix]))
                ix += 1

        self.set_free_parameters()


    @staticmethod
    def make_distribution(coordinates, name, arguments, lx):
        """
        Validate and construct a normalised distribution in the stated coordinates.
        FIXED entries have no continuous density and are excluded from sampling.
        """

        counts = dict(UNIFORM=2, LOGUNIFORM=2, GAUSSIAN=2, TRUNCGAUSSIAN=4, FIXED=1)
        if coordinates not in ('PHYSICAL', 'STATE'):
            raise ValueError('Coordinates must be PHYSICAL or STATE')
        if name not in counts:
            raise ValueError('Unknown prior distribution: ' + name)
        if len(arguments) != counts[name] or not np.all(np.isfinite(arguments)):
            raise ValueError(name + ' requires ' + str(counts[name]) + ' finite arguments')

        if name == 'FIXED':
            if coordinates == 'PHYSICAL' and lx == 1 and arguments[0] <= 0.0:
                raise ValueError('A logged parameter requires a positive physical fixed value')

            return None

        if name in ('UNIFORM', 'LOGUNIFORM'):
            lower, upper = arguments
            if lower >= upper or not np.isfinite(upper-lower):
                raise ValueError('Prior bounds must satisfy lower < upper with finite width')

            if name == 'LOGUNIFORM':
                if lower <= 0.0:
                    raise ValueError('LOGUNIFORM requires positive bounds')

                distribution = scipy.stats.loguniform(lower, upper)
            else:
                distribution = scipy.stats.uniform(lower, upper-lower)
        elif name == 'GAUSSIAN':
            mean, sigma = arguments
            if sigma <= 0.0:
                raise ValueError('Gaussian standard deviation must be positive')

            distribution = scipy.stats.norm(mean, sigma)
        else:
            mean, sigma, lower, upper = arguments
            if sigma <= 0.0 or lower >= upper:
                raise ValueError('TRUNCGAUSSIAN requires sigma > 0 and lower < upper')

            a, b = (lower-mean)/sigma, (upper-mean)/sigma
            if not np.all(np.isfinite([a, b])) or a >= b:
                raise ValueError('Truncated Gaussian bounds cannot be resolved numerically')

            distribution = scipy.stats.truncnorm(a, b, loc=mean, scale=sigma)

        if coordinates == 'PHYSICAL' and lx == 1 and distribution.support()[0] < 0.0:
            raise ValueError('A logged parameter requires positive physical support; use positive bounds or STATE coordinates')

        return distribution


    def read_nsp(self, runname):
        """
        Read optional runname.nsp overrides. The first non-comment line is
        NS_PRIORS 1. Subsequent lines contain:
        variable element PHYSICAL|STATE distribution arguments...

        Variable and element indices are one-based. Elements follow local
        state-vector order, not necessarily .apr line order. Blank lines and
        ! comments are ignored. Bounds/values are in the stated coordinates.
        """

        filename = os.fspath(runname) + '.nsp'

        try:
            f = open(filename)
        except FileNotFoundError:
            return

        header = False
        seen = set()

        with f:
            for line_number,line in enumerate(f, 1):
                fields = line.split('!', 1)[0].split()
                if not fields:
                    continue

                try:
                    if not header:
                        if fields != ['NS_PRIORS', '1']:
                            raise ValueError('Expected NS_PRIORS 1 header')

                        header = True
                        continue

                    if len(fields) < 5:
                        raise ValueError('Expected variable element coordinates distribution arguments')

                    ivar, element = int(fields[0]), int(fields[1])
                    if not 1 <= ivar <= len(self.NXVAR):
                        raise ValueError('Variable index is outside the .apr variable list')
                    if not 1 <= element <= self.NXVAR[ivar-1]:
                        raise ValueError('Element index is outside this variable state-vector block')

                    ix = int(self.NXVAR[:ivar-1].sum()) + element - 1
                    if ix in seen:
                        raise ValueError('Duplicate prior for this state-vector entry')

                    coordinates, name = fields[2:4]
                    arguments = [float(value) for value in fields[4:]]
                    distribution = self.make_distribution(coordinates, name, arguments, self.LX[ix])
                    self.DEFINITIONS[ix] = dict(variable=ivar, element=element, index=ix,
                                                coordinates=coordinates, distribution=name,
                                                arguments=arguments, LX=int(self.LX[ix]))
                    self.DISTRIBUTIONS[ix] = distribution

                    if name == 'FIXED':
                        self.XN[ix] = np.log(arguments[0]) if coordinates == 'PHYSICAL' and self.LX[ix] == 1 else arguments[0]

                    seen.add(ix)
                except (ValueError, OverflowError) as exception:
                    raise ValueError(filename + ':' + str(line_number) + ': ' + str(exception)) from exception

        if not header:
            raise ValueError(filename + ':1: Expected NS_PRIORS 1 header')

        self.HAS_FILE = True
        self.set_free_parameters()


    def set_free_parameters(self):
        self.vars_to_vary = [i for i,definition in enumerate(self.DEFINITIONS) if definition['distribution'] != 'FIXED']


    def check_shape(self, values):
        values = np.asarray(values, dtype=float)
        if values.ndim == 0 or values.shape[-1] != len(self.vars_to_vary) or not np.all(np.isfinite(values)):
            raise ValueError('Prior input must have one finite entry per free parameter')

        return values


    def from_physical(self, value, ix):
        if self.DEFINITIONS[ix]['coordinates'] == 'PHYSICAL' and self.LX[ix] == 1:
            if np.any(value <= 0.0):
                raise ValueError('Physical prior reached a non-positive value for a logged parameter')

            return np.log(value)

        return value


    def to_physical(self, value, ix):
        if self.DEFINITIONS[ix]['coordinates'] == 'PHYSICAL' and self.LX[ix] == 1:
            with np.errstate(over='ignore', under='ignore'):
                return np.exp(value)

        return value


    def Prior(self, cube):
        """
        Transform independent uniform coordinates to free Variables.XN entries.
        """

        cube = self.check_shape(cube)
        if np.any((cube <= 0.0) | (cube >= 1.0)):
            raise ValueError('Prior input must lie in the open unit cube')

        result = np.empty_like(cube)

        for i,ix in enumerate(self.vars_to_vary):
            result[...,i] = self.from_physical(self.DISTRIBUTIONS[ix].ppf(cube[...,i]), ix)

        return self.check_shape(result)


    def InversePrior(self, parameters):
        """
        Inverse scientific-prior transform for representable interior states.
        Sampling retains the generating latent coordinates separately.
        """

        return scipy.special.ndtr(self.ToNormal(parameters))


    def FromNormal(self, z):
        """
        Apply T(Phi(z)), retaining the analytic shortcut for Gaussian priors.
        Use the survival function in the upper tail, without clipping the cube.
        Finite precision can round a bounded quantile onto its endpoint. The
        sampler retains z separately and must use it for the density correction.
        """

        z = self.check_shape(z)
        result = np.empty_like(z)

        for i,ix in enumerate(self.vars_to_vary):
            definition = self.DEFINITIONS[ix]
            distribution = self.DISTRIBUTIONS[ix]

            if definition['distribution'] == 'GAUSSIAN':
                mean, sigma = definition['arguments']
                value = mean + sigma*z[...,i]
            elif definition['distribution'] == 'TRUNCGAUSSIAN':
                mean, sigma, lower, upper = definition['arguments']
                a, b = (lower-mean)/sigma, (upper-mean)/sigma

                # Difference of normal CDFs in log space, including tail intervals.
                if a >= 0.0:
                    large, small = scipy.special.log_ndtr(-a), scipy.special.log_ndtr(-b)
                else:
                    large, small = scipy.special.log_ndtr(b), scipy.special.log_ndtr(a)

                logmass = large + np.log(-np.expm1(small-large))
                logp = scipy.special.log_ndtr(z[...,i])
                logs = scipy.special.log_ndtr(-z[...,i])
                logcdf = np.logaddexp(scipy.special.log_ndtr(a), logp + logmass)
                logsf = np.logaddexp(scipy.special.log_ndtr(-b), logs + logmass)
                normal = np.where(
                    logcdf <= logsf,
                    scipy.special.ndtri_exp(logcdf),
                    -scipy.special.ndtri_exp(logsf)
                )
                value = mean + sigma*normal
                # Restore only floating-point overshoot of the known quantile support.
                value = np.minimum(np.maximum(value, lower), upper)

                if definition['coordinates'] == 'PHYSICAL' and self.LX[ix] == 1 and lower == 0.0:
                    # At an unresolved zero endpoint, F(x) = pdf(0)*x to machine
                    # precision. Keep log(x) finite without discarding this tail.
                    logx = logp - distribution.logpdf(0.0)
                    # Also handle cancellation leaving a tiny spurious positive
                    # value. The local CDF expansion is accurate to roundoff here.
                    logscale = np.log(sigma) - np.log1p(abs(mean/sigma))
                    unresolved = logx < logscale + np.log(np.finfo(float).eps)

                    with np.errstate(divide='ignore'):
                        result[...,i] = np.where(unresolved | (value == 0.0), logx, np.log(value))

                    continue
            else:
                probability = scipy.special.ndtr(-np.abs(z[...,i]))
                lower, upper = definition['arguments']

                if definition['coordinates'] == 'PHYSICAL' and self.LX[ix] == 1:
                    if definition['distribution'] == 'LOGUNIFORM':
                        width = np.log(upper)-np.log(lower)
                        result[...,i] = np.where(z[...,i] <= 0.0, np.log(lower)+width*probability,
                                                 np.log(upper)-width*probability)
                        continue

                    if lower == 0.0:
                        result[...,i] = np.log(upper) + scipy.special.log_ndtr(z[...,i])
                        continue

                value = np.where(z[...,i] <= 0.0, distribution.ppf(probability), distribution.isf(probability))
                value = np.minimum(np.maximum(value, lower), upper)

            result[...,i] = self.from_physical(value, ix)

        return self.check_shape(result)


    def ToNormal(self, parameters):
        """
        Phi^-1(T^-1(parameters)), using log-CDFs to preserve tail precision.
        Supports a single parameter vector or a matrix of posterior samples.
        """

        parameters = self.check_shape(parameters)
        result = np.empty_like(parameters)

        for i,ix in enumerate(self.vars_to_vary):
            definition = self.DEFINITIONS[ix]
            value = self.to_physical(parameters[...,i], ix)

            if definition['distribution'] == 'GAUSSIAN':
                mean, sigma = definition['arguments']
                result[...,i] = (value - mean) / sigma
            else:
                distribution = self.DISTRIBUTIONS[ix]
                logcdf, logsf = distribution.logcdf(value), distribution.logsf(value)
                result[...,i] = np.where(
                    logcdf <= logsf,
                    scipy.special.ndtri_exp(logcdf),
                    -scipy.special.ndtri_exp(logsf)
                )

        if not np.all(np.isfinite(result)):
            raise ValueError('Parameters are outside the prior interior or too close to a boundary to resolve')

        return result


    def MarginalLogPDF(self, i, values):
        """
        Density for free parameter i in state-vector coordinates, including
        the physical-to-log-coordinate Jacobian where LX == 1.
        """

        ix = self.vars_to_vary[i]
        values = np.asarray(values, dtype=float)
        density = self.DISTRIBUTIONS[ix].logpdf(self.to_physical(values, ix))

        if self.DEFINITIONS[ix]['coordinates'] == 'PHYSICAL' and self.LX[ix] == 1:
            density = density + values

        return density


    def LogPDF(self, parameters):
        parameters = self.check_shape(parameters)

        return sum(self.MarginalLogPDF(i, parameters[...,i]) for i in range(len(self.vars_to_vary)))
