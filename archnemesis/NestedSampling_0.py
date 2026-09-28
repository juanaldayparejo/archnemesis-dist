#!/usr/local/bin/python3
# -*- coding: utf-8 -*-
#
# archNEMESIS - Python implementation of the NEMESIS radiative transfer and retrieval code
# NestedSampling_0.py - Object to run nested sampling retrievals.
#
# Copyright (C) 2025 Juan Alday, Joseph Penn, Patrick Irwin,
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

#from archnemesis import *
import os
import sys
import json
import hashlib
import time
import copy
import functools
import traceback
import scipy
import scipy.linalg
import numpy as np
import corner
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

import archnemesis as ans
import archnemesis.Files
from archnemesis.NestedSamplingTransport_0 import NestedSamplingTransport_0, NestedSamplingMixture_0
from archnemesis.NestedSamplingPrior_0 import NestedSamplingPrior_0

import archnemesis.cfg.logs as logging
_lgr = logging.getLogger(__name__)
_lgr.setLevel(logging.DEBUG)

pymultinest = NotImplemented


class InvalidAtmosphericState(ValueError):
    """An explicitly identified physical state with zero scientific likelihood."""


class NestedSampling_0:
    

    def __init__(self, N_LIVE_POINTS=400, nemesisC=False):
        
        """
        Inputs
        ------
        @param N_LIVE_POINTS: int,
            Number of live points in retrieval 
        @param nemesisC: bool,
            Use the NEMESIS C forward model when True.

        Methods
        -------
        NestedSampling.reduced_chi_squared()
        NestedSampling.LogLikelihood()
        NestedSampling.Prior()
        NestedSampling.make_plots()
        """
        
        try:
            import pymultinest as _pymultinest # noqa F401
        except ImportError:
            _lgr.critical('PyMultiNest is not installed. Please download this before attempting to run retrievals with nested sampling. Instructions on installation can be found here: http://johannesbuchner.github.io/PyMultiNest/install.html')
            raise
        else:
            global pymultinest
            pymultinest = _pymultinest
        
        self.N_LIVE_POINTS = N_LIVE_POINTS
        self.nemesisC = nemesisC
        self.Emulator = None
        self.Transport = None
        self.PriorDistribution = None
        self.NORMALISE = False
        self.FULL_COVARIANCE = False
        self.N_FULL_CALLS = 0
        self.N_GUIDE_CALLS = 0
        self.N_OUTPUT_PARAMETERS = None
        self.StateValidator = None
        self.RUN_ID = None
        self.STAGE = None
        self.GUIDE_PREFIX = None
        self.TRANSPORT_SELECTION = None


    def valid_state(self, parameters):
        """
        Apply an optional scientific-domain check to the complete state vector.
        Only False or InvalidAtmosphericState signifies a physical rejection.
        Other failures are programming/configuration errors and must abort.
        """

        if self.StateValidator is None:
            return True

        state = self.PriorDistribution.XN.copy()
        state[self.vars_to_vary] = parameters

        try:
            valid = self.StateValidator(state)
        except InvalidAtmosphericState:
            return False

        if not isinstance(valid, (bool, np.bool_)):
            raise TypeError('NS_validate_state must return a boolean')

        return bool(valid)


    def measurement_vector(self, spectrum):
        """
        Pack geometries in Measurement.Y order, excluding padded channels.
        Emulators may return this vector directly or the usual spectral array.
        """

        spectrum = np.asarray(spectrum, dtype=float)

        if spectrum.ndim == 2:
            nconv = self.ForwardModel.Measurement.NCONV
            if spectrum.shape != (max(nconv), len(nconv)):
                raise ValueError('Spectrum must have shape (max(NCONV), NGEOM)')

            spectrum = np.concatenate([spectrum[:n,i] for i,n in enumerate(nconv)])

        if spectrum.shape != self.Y.shape or not np.all(np.isfinite(spectrum)):
            raise ValueError('Spectrum must contain one finite value per measurement')

        return spectrum


    def gaussian_setup(self, covariance):
        """
        Factor a fixed measurement covariance once for likelihood evaluation.
        """

        covariance = np.asarray(covariance, dtype=float)
        if covariance.shape != (len(self.Y), len(self.Y)) or not np.all(np.isfinite(covariance)):
            raise ValueError('Likelihood covariance must be a finite NY by NY matrix')
        if not np.allclose(covariance, covariance.T, rtol=1.0e-10, atol=0.0):
            raise ValueError('Likelihood covariance must be symmetric')

        if np.array_equal(covariance, np.diag(covariance.diagonal())):
            if np.any(covariance.diagonal() <= 0.0):
                raise ValueError('Measurement variances must be positive')

            factor = np.sqrt(covariance.diagonal())
            logdet = 2.0 * np.log(factor).sum()
        else:
            factor = np.linalg.cholesky(covariance)
            logdet = 2.0 * np.log(factor.diagonal()).sum()

        constant = -0.5 * (len(self.Y) * np.log(2.0 * np.pi) + logdet)

        return factor, constant


    def gaussian_loglikelihood(self, residual, setup):
        """
        Evaluate the quadratic and, when requested, Gaussian normalisation.
        """

        factor, constant = setup

        if factor.ndim == 1:
            residual = residual / factor
        else:
            residual = scipy.linalg.solve_triangular(factor, residual, lower=True)

        value = -0.5 * np.dot(residual, residual) + (constant if self.NORMALISE else 0.0)
        if not np.isfinite(value):
            raise ValueError('Gaussian likelihood is not finite')

        return value


    def chi_squared(self, a,b,err):
        """
        Calculate chi^2/n statistic.
        """        

        return np.sum(((a - b.T.flatten())**2)/(err**2))


    def LogLikelihood(self,cube):
        """
        Compute likelihood - run a forward model and compare to spectrum.
        """   
        
        if not self.valid_state(cube):
            return -np.inf

        if self.PriorDistribution is not None:
            self.ForwardModel.Variables.XN[:] = self.PriorDistribution.XN

        self.ForwardModel.Variables.XN[self.vars_to_vary] = cube
        self.N_FULL_CALLS += 1
        
        original_stdout = sys.stdout  

        try:
            sys.stdout = open(os.devnull, 'w')  # Redirect stdout

            if hasattr(self, 'ForwardModelFunction'):
                YN = self.ForwardModelFunction()
            elif self.nemesisC:
                YN = self.ForwardModel.nemesisCfm()
            else:
                YN = self.ForwardModel.nemesisfm()
        except InvalidAtmosphericState:
            return -np.inf
        finally:
            sys.stdout.close()  # Close the devnull
            sys.stdout = original_stdout  # Restore the original stdout
        
        if not hasattr(self, 'LIKELIHOOD_SETUP'):
            return -self.chi_squared(self.Y,YN,self.Y_ERR)/2

        residual = self.Y - self.measurement_vector(YN)

        return self.gaussian_loglikelihood(residual, self.LIKELIHOOD_SETUP)


    def GuideLogLikelihood(self, cube):
        """
        Error-aware emulator likelihood; epsilon = emulator - full model.
        The emulator receives free parameters in Variables.XN coordinates.
        """

        if not self.valid_state(cube):
            return -np.inf

        self.N_GUIDE_CALLS += 1
        spectrum = self.measurement_vector(self.Emulator(np.array(cube, copy=True)))
        residual = self.Y - spectrum + self.EMULATOR_MEAN

        return self.gaussian_loglikelihood(residual, self.GUIDE_SETUP)


    def AuxiliaryPrior(self, cube):
        """
        Compose T and S using the resolved scientific priors.
        MultiNest output remains in the original state-vector coordinates.
        """

        z = self.Transport.Prior(cube)

        return self.PriorDistribution.FromNormal(z)


    def RepartitionedLogLikelihood(self, cube, z):
        """
        Full-model log likelihood plus the exact fitted density correction.
        """

        return self.LogLikelihood(cube) + self.Transport.LogCorrection(z)
    

    def Prior(self, cube):
        """
        Map unit cube to prior distributions.
        """  
        
        if self.PriorDistribution is not None:
            return self.PriorDistribution.Prior(cube)

        cube1 = cube.copy()

        for i in range(len(self.vars_to_vary)):
              cube1[i] = self.priors[i](cube1[i])

        return cube1


    def InversePrior(self, parameters):
        """
        Inverse of the resolved scientific-prior transform.
        """

        if self.PriorDistribution is not None:
            return self.PriorDistribution.InversePrior(parameters)

        z = (np.asarray(parameters) - self.XA[self.vars_to_vary]) / self.XA_ERR[self.vars_to_vary]

        return scipy.special.ndtr(z)


    def get_diagnostics(self, stage=None):
        """
        Return weighted samples in state-vector and sampling coordinates.
        stage defaults to the returned stage; 'guide' also works after a full run.
        transformed_coordinates are S^-1(z), using the frozen fitted transport.
        Fitted log densities are in state-vector coordinates, not physical units.
        This method reads sampler output and never evaluates a forward model.
        """

        stage = self.STAGE if stage is None else stage
        if stage not in ('guide', 'full'):
            raise ValueError("Diagnostic stage must be 'guide' or 'full'")

        if stage == 'guide':
            if self.GUIDE_PREFIX is None:
                raise ValueError('This run has no emulator guide')

            prefix = self.GUIDE_PREFIX
            result = self.guide_result
        else:
            if self.STAGE != 'full':
                raise ValueError('The full-model stage has not been run')

            prefix = self.prefix
            result = self.result

        data = _ns_analyzer(self, prefix).get_data()
        data = data[_ns_posterior_mask(data)]
        ndim = len(self.parameters)
        samples = data[:,2:2+ndim].copy()
        z = data[:,2+ndim:2+2*ndim].copy()
        weights = data[:,0] / data[:,0].sum()
        diagnostics = dict(stage=stage, samples=samples, weights=weights,
                           latent_coordinates=z, prior_coordinates=scipy.special.ndtr(z),
                           vars_to_vary=list(self.vars_to_vary),
                           priors=copy.deepcopy(self.PriorDistribution.DEFINITIONS),
                           log_likelihood=data[:,2+2*ndim].copy(),
                           log_correction=data[:,3+2*ndim].copy(),
                           logZ=result['logZ'], logZerr=result['logZerr'],
                           effective_samples=float(1.0/np.sum(weights**2)),
                           transport=None, selection=copy.deepcopy(self.TRANSPORT_SELECTION))

        if self.Transport is not None:
            diagnostics['transport'] = self.Transport.coefficients()
            diagnostics['transformed_coordinates'] = np.array([self.Transport.InversePrior(point) for point in z])

            if isinstance(self.Transport, NestedSamplingMixture_0):
                logdensity = self.Transport.LogDensity(z)
            else:
                white = scipy.linalg.solve_triangular(self.Transport.CHOLESKY, (z-self.Transport.MEAN).T, lower=True).T
                logdensity = scipy.stats.norm.logpdf(white).sum(axis=1) - self.Transport.LOGDET

            diagnostics['fitted_log_density'] = (logdensity + self.PriorDistribution.LogPDF(samples)
                                                 - scipy.stats.norm.logpdf(z).sum(axis=1))

        return diagnostics


    def make_diagnostic_plots(self, stage=None, parameters=None, labels=None, truth=None,
                              full_prior=True, output_directory=None):
        """
        Save posterior/prior/fitted-density and transformed-space corner plots.
        parameters selects zero-based state-vector indices; labels follows that order.
        truth, when supplied, contains all free entries in Variables.XN coordinates.
        Logged parameters are therefore plotted in natural-log parameter space.
        Bounded priors use their full support by default; unbounded priors use
        their 0.001--0.999 quantiles, expanded to include the posterior and truth.
        Fitted contours use reproducible mixture draws; returned densities from
        get_diagnostics are evaluated directly. Returns the saved file paths.
        """

        diagnostics = self.get_diagnostics(stage)
        ndim = len(self.parameters)
        selected = list(self.vars_to_vary) if parameters is None else list(parameters)
        if not selected or len(set(selected)) != len(selected) or any(ix not in self.vars_to_vary for ix in selected):
            raise ValueError('parameters must select distinct free state-vector indices')

        columns = [list(self.vars_to_vary).index(ix) for ix in selected]

        if labels is None:
            labels = [('ln(parameter %d)' if self.PriorDistribution.LX[ix] == 1 else 'parameter %d') % ix for ix in selected]

        if len(labels) != len(columns):
            raise ValueError('Supply one label per selected parameter')

        if truth is not None:
            truth = np.asarray(truth, dtype=float)
            if truth.shape != (ndim,) or not np.all(np.isfinite(truth)):
                raise ValueError('truth must contain every free state-vector entry')

        prefix = self.GUIDE_PREFIX if diagnostics['stage'] == 'guide' else self.prefix
        directory = prefix if output_directory is None else os.fspath(output_directory)
        os.makedirs(directory, exist_ok=True)
        samples, weights = diagnostics['samples'], diagnostics['weights']
        ranges = []

        for i,ix in zip(columns, selected):
            order = np.argsort(samples[:,i])
            limits = np.interp([0.001, 0.999], np.cumsum(weights[order])-0.5*weights[order], samples[order,i])

            if full_prior:
                distribution = self.PriorDistribution.DISTRIBUTIONS[ix]
                support = np.array(distribution.support())
                quantiles = distribution.ppf([0.001, 0.999])
                definition = self.PriorDistribution.DEFINITIONS[ix]

                if definition['coordinates'] == 'PHYSICAL' and definition['LX'] == 1:
                    with np.errstate(divide='ignore'):
                        support, quantiles = np.log(support), np.log(quantiles)

                limits = np.array([support[0] if np.isfinite(support[0]) else min(quantiles[0], limits[0]),
                                   support[1] if np.isfinite(support[1]) else max(quantiles[1], limits[1])])

            if truth is not None:
                limits = np.array([min(limits[0], truth[i]), max(limits[1], truth[i])])

            if not limits[0] < limits[1]:
                limits += np.array([-1.0, 1.0])*max(abs(limits[0]), 1.0)*1.0e-6

            ranges.append(tuple(limits))

        truths = None if truth is None else truth[columns]
        figure = corner.corner(samples[:,columns], weights=weights, labels=labels, range=ranges,
                               truths=truths, truth_color='black', color='tab:blue', bins=40,
                               levels=(0.68, 0.95), plot_datapoints=False, hist_kwargs={'density': True})
        legend = [Line2D([], [], color='tab:blue', label=diagnostics['stage'].capitalize()+' posterior'),
                  Line2D([], [], color='tab:red', label='Scientific prior')]

        if self.Transport is not None:
            rng = np.random.default_rng(0)
            coefficients = diagnostics['transport']
            means, factors = np.asarray(coefficients['MEAN']), np.asarray(coefficients['CHOLESKY'])

            if means.ndim == 1:
                means, factors = means[None,:], factors[None,:,:]

            mixing = np.asarray(coefficients.get('WEIGHTS', [1.0]))
            component = rng.choice(len(means), size=10000, p=mixing)
            z = means[component] + np.einsum('nij,nj->ni', factors[component], rng.standard_normal((10000, ndim)))
            fitted = self.PriorDistribution.FromNormal(z)

            if len(columns) == 1:
                figure.axes[0].hist(fitted[:,columns[0]], bins=40, range=ranges[0],
                                    density=True, histtype='step', color='tab:purple')
            else:
                corner.corner(fitted[:,columns], fig=figure, range=ranges, color='tab:purple', bins=40,
                              levels=(0.68, 0.95), plot_datapoints=False, fill_contours=False,
                              hist_kwargs={'density': True})

            legend.append(Line2D([], [], color='tab:purple', label='Fitted density (draws)'))

        axes = np.asarray(figure.axes).reshape(len(columns), len(columns))

        for j,i in enumerate(columns):
            x = np.linspace(*ranges[j], 1000)
            axes[j,j].plot(x, np.exp(self.PriorDistribution.MarginalLogPDF(i, x)), color='tab:red')

        figure.legend(handles=legend, loc='upper right')
        paths = {}
        def save(figure, name):
            for extension in ('png', 'pdf'):
                filename = os.path.join(directory, name+'.'+extension)
                figure.savefig(filename, dpi=150)
                paths[name+'.'+extension] = filename

            plt.close(figure)
        save(figure, 'posterior_diagnostics')

        if self.Transport is not None:
            cube = diagnostics['transformed_coordinates']
            transformed_truth = None if truth is None else self.Transport.InversePrior(self.PriorDistribution.ToNormal(truth))[columns]
            figure = corner.corner(cube[:,columns], weights=weights, range=[(0.0, 1.0)]*len(columns),
                                   labels=['v%d' % (i+1) for i in columns], truths=transformed_truth,
                                   truth_color='black', color='tab:blue', bins=40, levels=(0.68, 0.95),
                                   plot_contours=False, plot_density=False, plot_datapoints=False,
                                   hist_kwargs={'density': True})
            axes = np.asarray(figure.axes).reshape(len(columns), len(columns))

            for j in range(len(columns)):
                axes[j,j].axhline(1.0, color='tab:purple', linestyle='--')

                for k in range(j):
                    axes[j,k].scatter(cube[:,columns[k]], cube[:,columns[j]],
                                      s=20.0*weights/weights.max(), alpha=0.3,
                                      color='tab:blue', edgecolors='none', rasterized=True)

            figure.suptitle(diagnostics['stage'].capitalize()+' posterior after fitted transformation; fitted density = 1')
            save(figure, 'transformed_diagnostics')

        if self.TRANSPORT_SELECTION is not None:
            scores = [row for row in self.TRANSPORT_SELECTION['scores'] if 'mean' in row]
            figure, ax = plt.subplots()
            ax.errorbar([row['components'] for row in scores], [row['mean'] for row in scores],
                        yerr=[row['standard_error'] for row in scores], fmt='o-', capsize=4)
            ax.axvline(self.TRANSPORT_SELECTION['selected_components'], color='tab:purple', linestyle='--')
            ax.set(xlabel='Gaussian components', ylabel='Held-out weighted log density',
                   title='Transport component selection (fold variation)')
            save(figure, 'component_selection')

        return paths
    

    def make_plots(self):
        """
        Cornerplot of results with analytical prior.
        """

        prior_means = self.XA
        prior_stds = self.XA_ERR

        # Initialize the analyzer
        a = _ns_analyzer(self)
        #s = a.get_stats()

        _lgr.info('Creating marginal plot ...')

        # Extract data and weights
        data_array = a.get_data()
        weights = data_array[:, 0]
        data = data_array[:, 2:2+len(self.parameters)]

        # Retain every positive-weight posterior row, regardless of sample count.
        mask = _ns_posterior_mask(data_array)
        data_masked = data[mask, :]
        weights_masked = weights[mask]
        weights_masked = weights_masked / weights_masked.sum()

        # Determine axis ranges from posterior samples
        ranges = []

        for i in range(len(self.parameters)):
            min_val = np.nanmin(data_masked[:, i])
            max_val = np.nanmax(data_masked[:, i])
            ranges.append((min_val-0.01, max_val+0.01))

        # Plot posterior samples
        figure = corner.corner(
            data_masked,
            weights=weights_masked,
            labels=self.parameters,
            show_titles=True,
            color='blue',
            range=ranges,
            bins=50,  # Adjust as needed
            hist_kwargs={'density': True},
            plot_contours=True,
            fill_contours=False,
            contour_colors=['blue'],
            smooth=1.0,
            data_kwargs={'alpha': 0.5},  # Adjust transparency
        )

        # Overlay analytical prior
        axes = np.array(figure.axes).reshape((len(self.parameters), len(self.parameters)))

        for i in range(len(self.parameters)):
            x = np.linspace(ranges[i][0], ranges[i][1], 1000)

            if self.PriorDistribution is not None:
                y = np.exp(self.PriorDistribution.MarginalLogPDF(i, x))
            else:
                y = scipy.stats.norm(prior_means[self.vars_to_vary[i]], prior_stds[self.vars_to_vary[i]]).pdf(x)

            ax = axes[i, i]
            ax.plot(x, y, color='red', lw=2, label='Prior')

        # Add legends
        from matplotlib.lines import Line2D
        legend_elements = [
            Line2D([0], [0], color='blue', lw=2, label='Posterior'),
            Line2D([0], [0], color='red', lw=2, label='Prior')
        ]
        figure.legend(handles=legend_elements, loc='upper right')

        plt.savefig(self.prefix + 'corner.png')
        plt.close()


    def extract(self):
        """
        Extracts the fitted parameter values and their uncertainties.

        Returns:
        --------
        dict
            A dictionary with parameter names as keys and tuples of 
            (mean, standard deviation) as values.
        """

        if not hasattr(self, 'result') or 'samples' not in self.result:
            raise AttributeError("No results found. Ensure the sampling process has been completed.")

        # Extract parameter samples
        samples = self.result['samples']
        parameters = self.parameters

        # Compute mean and standard deviation for each parameter
        parameter_values = {
            param: (samples[:, i].mean(), samples[:, i].std())
            for i, param in enumerate(parameters)
        }

        return parameter_values        
        

    def compare(self):
        """
        Plots a corner plot of the current run's posterior samples and compares them
        to Gaussian distributions derived from retprof and reterr from a .mre file.

        Parameters
        ----------
        """

        lat, lon, ngeom, ny, wave, specret, specmeas, specerrmeas, nx, Var, aprprof, aprerr, retprof, reterr = ans.Files.read_mre(self.ForwardModel.runname)

        # Load posterior samples from the current run
        analyzer = _ns_analyzer(self)
        data_array = analyzer.get_data()
        weights = data_array[:, 0]
        samples = data_array[:, 2:2+len(self.parameters)]

        mask = _ns_posterior_mask(data_array)
        samples_masked = samples[mask, :]
        weights_masked = weights[mask]

        # Load the covariance matrix for optimal estimation
        full_covariance_matrix = ans.Files.read_cov(self.ForwardModel.runname)[9]

        # Extract indices for the parameters of interest
        parameter_indices = [int(ip) for ip in self.parameters]

        # Extract the relevant submatrix corresponding to the parameters
        covariance_matrix = full_covariance_matrix[np.ix_(parameter_indices, parameter_indices)]

        # Compute the mean vector in log-space
        mean_vector = np.log(retprof[parameter_indices, 0])

        try:
            # Attempt Cholesky decomposition to check positive definiteness
            np.linalg.cholesky(covariance_matrix)
        except np.linalg.LinAlgError:
            # If not positive definite, add a small value to the diagonal
            epsilon = 1e-10
            covariance_matrix += epsilon * np.eye(len(self.parameters))
            _lgr.info("Covariance matrix was not positive definite. Added small epsilon to diagonal.")

        # Generate Gaussian samples using the covariance matrix
        num_gaussian_samples = 100000
        gaussian_samples = np.random.multivariate_normal(mean=mean_vector, cov=covariance_matrix, size=num_gaussian_samples)

        # Create a corner plot for the nested sampling posterior
        figure = corner.corner(
            samples_masked,
            weights=weights_masked,
            labels=self.parameters,
            color="blue",
            bins=50,
            hist_kwargs={'density': True},
            show_titles=True,
            plot_contours=True,
            fill_contours=False,
            contour_colors=["blue"],
            title_fmt=".2f",
            smooth=1.0
        )

        # Overlay the Gaussian distributions from optimal estimation
        corner.corner(
            gaussian_samples,
            labels=self.parameters,
            color="red",
            bins=50,
            hist_kwargs={'density': True},
            show_titles=False,
            plot_contours=True,
            fill_contours=False,
            plot_datapoints=False,  # Hide individual points for clarity
            contour_colors=["red"],
            fig=figure,
            smooth=1.0
        )

        # Add a legend to differentiate between the two posteriors
        legend_elements = [
            Line2D([0], [0], color="blue", lw=2, label="Nested Sampling Posterior"),
            Line2D([0], [0], color="red", lw=2, label="Optimal Estimation Result"),
        ]
        figure.legend(handles=legend_elements, loc="upper right")

        plt.show()
        
def _ns_rank_zero(comm, function):
    """
    Complete file operations and transport fitting on rank zero.
    Broadcast failures as well as results so other ranks do not wait forever.
    """

    result = None
    error = None

    if comm.Get_rank() == 0:
        try:
            result = function()
        except Exception as exception:
            error = str(exception)

    error = comm.bcast(error, root=0)
    if error is not None:
        raise RuntimeError(error)

    return comm.bcast(result, root=0)


def _ns_abort(comm):
    """
    A Python exception must not escape a ctypes callback and leave MPI waiting.
    Print the original traceback, terminate all ranks, and never return to C.
    """

    try:
        traceback.print_exc(file=sys.stderr)
        sys.stderr.flush()
    finally:
        try:
            comm.Abort(1)
        finally:
            os._exit(1)


def _ns_mpi_errors(function):
    """Stop other MPI ranks if one rank fails outside the sampler callbacks."""
    @functools.wraps(function)

    def wrapped(*args, **kwargs):
        from mpi4py import MPI

        try:
            return function(*args, **kwargs)
        except BaseException:
            if MPI.COMM_WORLD.Get_size() > 1:
                _ns_abort(MPI.COMM_WORLD)

            raise

    return wrapped


def _ns_analyzer(NestedSampling, prefix=None):
    nparams = NestedSampling.N_OUTPUT_PARAMETERS

    if nparams is None:
        nparams = len(NestedSampling.parameters)

    return pymultinest.Analyzer(n_params=nparams,
                               outputfiles_basename=NestedSampling.prefix if prefix is None else prefix)


def _ns_posterior_mask(data):
    """Keep every positive-weight row; zero-weight rows need no finite values."""

    data = np.asarray(data)
    if data.ndim != 2 or data.shape[1] < 3:
        raise ValueError('Expected a weighted MultiNest posterior matrix')

    weights = data[:,0]
    if not np.all(np.isfinite(weights)) or np.any(weights < 0.0):
        raise ValueError('Posterior weights must be finite and non-negative')

    keep = weights > 0.0
    if not np.any(keep) or not np.all(np.isfinite(data[keep])):
        raise ValueError('Posterior requires finite positive-weight samples')

    return keep


def _ns_run(NestedSampling, comm, prefix, options, guide=False):
    """
    Retain the generating latent coordinates instead of inverting rounded states.
    Native output columns after weight and -2 log likelihood are: free XN (d),
    latent z (d), uncorrected log likelihood (1), and log correction (1).
    Only the first d coordinates are sampled or used for mode clustering.
    """

    ndim = len(NestedSampling.parameters)
    NestedSampling.N_OUTPUT_PARAMETERS = 2*ndim + 2

    def likelihood(cube, n_dims, n_params):
        try:
            u = np.array([cube[i] for i in range(ndim)])
            if not np.all(np.isfinite(u)) or np.any((u < 0.0) | (u > 1.0)):
                raise ValueError('MultiNest proposal is outside the unit cube')

            # Exact endpoints have zero measure. Move only these endpoints to
            # their nearest interior float; never clip an interval of the prior.
            u[u == 0.0] = np.nextafter(0.0, 1.0)
            u[u == 1.0] = np.nextafter(1.0, 0.0)
            transport = None if guide else NestedSampling.Transport
            z = scipy.special.ndtri(u) if transport is None else transport.Prior(u)
            parameters = NestedSampling.PriorDistribution.FromNormal(z)
            correction = 0.0 if transport is None else transport.LogCorrection(z)
            value = NestedSampling.GuideLogLikelihood(parameters) if guide else NestedSampling.LogLikelihood(parameters)
            if not np.isfinite(correction) or np.isnan(value) or value == np.inf:
                raise ValueError('Non-finite nested sampling likelihood or density correction')

            total = value + correction
            if value != -np.inf and not np.isfinite(total):
                raise ValueError('Non-finite corrected nested sampling likelihood')

            for i,entry in enumerate(np.concatenate((parameters, z, [value, correction]))):
                cube[i] = entry

            return -1.0e100 if value == -np.inf else float(total)
        except BaseException:
            _ns_abort(comm)

    def run_sampler():
        pymultinest.run(LogLikelihood=likelihood, Prior=None, n_dims=ndim,
                       n_params=NestedSampling.N_OUTPUT_PARAMETERS,
                       outputfiles_basename=prefix, use_MPI=not guide, **options)

    if guide:
        # Use the serial native library on rank zero; other ranks wait here.
        _ns_rank_zero(comm, run_sampler)
    else:
        run_sampler()

    return _ns_rank_zero(comm, lambda: _ns_read_result(NestedSampling, prefix))


def _ns_read_result(NestedSampling, prefix):
    """Read an existing stage without evaluating its likelihood again."""

    analyzer = _ns_analyzer(NestedSampling, prefix)
    stats = analyzer.get_stats()
    samples = np.atleast_2d(analyzer.get_equal_weighted_posterior())

    return dict(logZ=stats['nested sampling global log-evidence'],
                logZerr=stats['nested sampling global log-evidence error'],
                samples=samples[:,:len(NestedSampling.parameters)])


def _ns_read_guide_setup(prefix, errors=False):
    """Load the saved guide settings or its fixed emulator-error arrays."""

    if errors:
        with np.load(os.path.join(prefix, 'emulator_error.npz')) as data:
            return data['mean'].copy(), data['covariance'].copy()

    with open(os.path.join(prefix, 'repartitioning.json')) as f:
        return json.load(f)


def _ns_write_guide_setup(NestedSampling, prefix):
    """Retain error inputs so a separate full run need not load the emulator."""

    filename = os.path.join(prefix, 'emulator_error.npz')

    with open(filename+'.tmp', 'wb') as f:
        np.savez(f, mean=NestedSampling.EMULATOR_MEAN, covariance=NestedSampling.EMULATOR_COV)

    os.replace(filename+'.tmp', filename)


def _ns_save_json(filename, value):
    """Commit a complete checkpoint with one atomic replacement."""

    with open(filename + '.tmp', 'w') as f:
        json.dump(value, f, indent=4, allow_nan=False)
        f.flush()
        os.fsync(f.fileno())

    os.replace(filename + '.tmp', filename)


def _ns_transport_hash(coefficients):
    return hashlib.sha256(json.dumps(coefficients, sort_keys=True, allow_nan=False).encode()).hexdigest()


def _ns_options(options, guided=False, guide=False):
    """
    Sampler options that do not replace callbacks, dimensions or output paths.
    """

    result = dict(n_live_points=1000 if guide else 400,
                  evidence_tolerance=0.1 if guide else 0.5,
                  resume=not guided, verbose=True)
    allowed = {'n_live_points', 'evidence_tolerance', 'sampling_efficiency',
               'seed', 'resume', 'verbose', 'n_iter_before_update',
               'importance_nested_sampling', 'multimodal', 'const_efficiency_mode'}

    if options is not None:
        unknown = set(options) - allowed
        if unknown:
            raise ValueError('Unsupported nested sampling options: ' + ', '.join(sorted(unknown)))

        result.update(options)

    if not isinstance(result['n_live_points'], (int, np.integer)) or result['n_live_points'] < 2:
        raise ValueError('n_live_points must be an integer greater than one')
    if not np.isfinite(result['evidence_tolerance']) or result['evidence_tolerance'] <= 0.0:
        raise ValueError('evidence_tolerance must be finite and positive')

    return result


def _ns_signature(NestedSampling, Variables, Measurement, runname, NS_run_id, options, guide_options, modes):
    """
    Identify the likelihood, prior and sampler settings for a guided restart.
    NS_run_id identifies external emulator weights and forward-model inputs.
    """

    digest = hashlib.sha256()

    for value in (NestedSampling.XA, NestedSampling.XA_ERR, Variables.XN,
                  Variables.LX, Variables.VARIDENT, Variables.VARPARAM,
                  NestedSampling.vars_to_vary, Measurement.Y, Measurement.SE,
                  Measurement.NCONV, Measurement.VCONV,
                  NestedSampling.EMULATOR_MEAN, NestedSampling.EMULATOR_COV):
        if value is None:
            digest.update(b'None')
            continue

        value = np.ascontiguousarray(value)
        digest.update(str(value.dtype).encode())
        digest.update(str(value.shape).encode())
        digest.update(value.tobytes())

    signature = dict(version=3, runname=os.path.abspath(runname), run_id=NS_run_id,
                input_hash=digest.hexdigest(), normalise=NestedSampling.NORMALISE,
                full_covariance=NestedSampling.FULL_COVARIANCE,
                state_validator=NestedSampling.StateValidator is not None,
                guide_mpi=False,
                output_parameters=NestedSampling.N_OUTPUT_PARAMETERS,
                priors=NestedSampling.PriorDistribution.DEFINITIONS,
                modes=list(modes),
                options={key:value for key,value in options.items() if key not in ('resume', 'verbose')},
                guide_options={key:value for key,value in guide_options.items() if key not in ('resume', 'verbose')})

    if NestedSampling.TRANSPORT_OPTIONS is not None:
        signature['transport_options'] = NestedSampling.TRANSPORT_OPTIONS

    return json.loads(json.dumps(signature, default=lambda value: value.item()))


def _ns_prepare_transport(prefix, signature, resume, stage='both'):
    """
    Prepare a fresh guided run or recover its unchanged, frozen transport.
    """

    manifest_file = os.path.join(prefix, 'repartitioning.json')
    transport_file = os.path.join(prefix, 'transport.npz')

    if resume or stage == 'full':
        if not signature['run_id']:
            raise ValueError('Guided resume requires the NS_run_id used for the original run')

        with open(manifest_file) as f:
            manifest = json.load(f)

        if manifest['signature'] != signature:
            raise ValueError('Guided resume settings or inputs differ from the saved run')

        coefficients = manifest.get('transport')

        if coefficients is not None:
            if manifest.get('transport_hash') != _ns_transport_hash(coefficients):
                raise ValueError('Saved transport does not match the guided run manifest')

            NestedSamplingTransport_0.from_coefficients(coefficients)
            if stage == 'full' and not resume and os.listdir(os.path.join(prefix, 'full')):
                raise ValueError('Full-model output already exists; use resume=True')

            return coefficients

        if stage == 'full':
            raise ValueError('The full stage requires a completed guide and saved transport')
        if manifest.get('transport_hash') is not None or os.listdir(os.path.join(prefix, 'full')):
            raise ValueError('Cannot resume the full-model run without its saved transport')
    else:
        if os.path.exists(manifest_file) or os.path.exists(transport_file):
            raise ValueError('Guided output already exists; use resume=True or a new NS_prefix')

        for directory in ('guide', 'full'):
            path = os.path.join(prefix, directory)
            if os.path.exists(path) and os.listdir(path):
                raise ValueError('Guided output directory is not empty: ' + path)

            os.makedirs(path, exist_ok=True)

        _ns_save_json(manifest_file, dict(signature=signature))

    return None


def _ns_fit_transport(NestedSampling, prefix, signature):
    """
    Fit the weighted guide and save its transport before any slow sampling.
    """

    analyzer = _ns_analyzer(NestedSampling, os.path.join(prefix, 'guide', ''))
    data = analyzer.get_data()
    data = data[_ns_posterior_mask(data)]
    ndim = len(NestedSampling.parameters)
    z = data[:,2+ndim:2+2*ndim]
    settings = NestedSampling.TRANSPORT_OPTIONS

    if settings is None:
        Transport = NestedSamplingTransport_0.from_samples(z, data[:,0])
    else:
        Transport = NestedSamplingMixture_0.from_samples(z, data[:,0],
                      max_components=settings['max_components'], seed=settings['seed'])

    coefficients = Transport.coefficients()
    manifest_file = os.path.join(prefix, 'repartitioning.json')
    _ns_save_json(manifest_file, dict(signature=signature, transport=coefficients,
                                    transport_hash=_ns_transport_hash(coefficients),
                                    selection=getattr(Transport, 'SELECTION', None)))

    return coefficients


def _ns_check_emulator(NestedSampling):
    """
    Return a validation error for collective checking before sampler callbacks.
    """

    try:
        parameters = NestedSampling.Prior(np.full(len(NestedSampling.vars_to_vary), 0.5))
        if not NestedSampling.valid_state(parameters):
            # The median need not lie inside the scientific physical domain.
            # Validate predictions when the first admissible proposal arrives.
            return None

        value = NestedSampling.GuideLogLikelihood(parameters)
        if not np.isfinite(value):
            raise ValueError('Guide likelihood is not finite at the prior median')
    except Exception as exception:
        return str(exception)

    return None


def _ns_write_result(NestedSampling, Variables, prefix=None, result=None, stage=None):
    """
    Save posterior weights and distinguish true and repartitioned likelihoods.
    Counts and wall time refer to this invocation, including any guide stage.
    """

    prefix = NestedSampling.prefix if prefix is None else prefix
    result = NestedSampling.result if result is None else result
    stage = NestedSampling.STAGE if stage is None else stage
    analyzer = _ns_analyzer(NestedSampling, prefix)
    data = analyzer.get_data()
    _ns_posterior_mask(data)
    ndim = len(NestedSampling.parameters)
    correction = data[:,3+2*ndim]
    filename = prefix + 'posterior.npz'

    with open(filename + '.tmp', 'wb') as f:
        np.savez(f, samples=data[:,2:2+ndim], weights=data[:,0],
                 latent_coordinates=data[:,2+ndim:2+2*ndim],
                 equal_weighted_samples=result['samples'],
                 sampling_log_likelihood=-0.5*data[:,1],
                 log_likelihood=data[:,2+2*ndim], log_correction=correction,
                 logZ=result['logZ'], logZerr=result['logZerr'], stage=stage,
                 vars_to_vary=NestedSampling.vars_to_vary, LX=Variables.LX[NestedSampling.vars_to_vary],
                 priors=json.dumps(NestedSampling.PriorDistribution.DEFINITIONS),
                 normalise=NestedSampling.NORMALISE, full_covariance=NestedSampling.FULL_COVARIANCE,
                 full_calls=NestedSampling.N_FULL_CALLS, guide_calls=NestedSampling.N_GUIDE_CALLS,
                 elapsed_seconds=NestedSampling.ELAPSED_SECONDS)

    os.replace(filename + '.tmp', filename)


def _ns_read_priors(Variables, runname):
    """
    Read optional nested-sampling overrides without changing Variables.
    """

    Priors = NestedSamplingPrior_0(Variables)
    Priors.read_nsp(runname)
    if not Priors.vars_to_vary:
        raise ValueError('Nested sampling requires at least one free parameter')

    for definition in Priors.DEFINITIONS:
        _lgr.info('NS prior: variable %d element %d -> state index %d (LX=%d): %s %s %s',
                  definition['variable'], definition['element'], definition['index'], definition['LX'],
                  definition['coordinates'], definition['distribution'], definition['arguments'])

    return Priors


def _ns_prepare_priors(prefix, Priors, resume):
    """
    Guard ordinary-run restarts with all resolved priors, including .apr defaults.
    """

    filename = prefix + 'priors.json'

    if resume:
        if os.path.exists(filename):
            with open(filename) as f:
                saved = json.load(f)

            if saved != Priors.DEFINITIONS:
                raise ValueError('Prior definitions differ from the saved ordinary run; use a new NS_prefix')

            return

        if any(os.path.exists(prefix + suffix) for suffix in ('.txt', 'resume.dat', 'live.points', 'phys_live.points')):
            raise ValueError('Cannot resume existing chains without saved prior definitions; use a new NS_prefix')

    _ns_save_json(filename, Priors.DEFINITIONS)


def _ns_prepare_likelihood(prefix, NestedSampling, covariance, resume):
    """
    Record the ordinary-run likelihood and prevent a restart with a changed
    normalisation, covariance choice or set of measurements.
    """

    digest = hashlib.sha256()

    for value in (NestedSampling.Y, covariance):
        value = np.ascontiguousarray(value, dtype=float)
        digest.update(str(value.shape).encode())
        digest.update(value.tobytes())

    signature = dict(version=2, normalise=NestedSampling.NORMALISE,
                     full_covariance=NestedSampling.FULL_COVARIANCE,
                     run_id=NestedSampling.RUN_ID,
                     state_validator=NestedSampling.StateValidator is not None,
                     output_parameters=NestedSampling.N_OUTPUT_PARAMETERS,
                     input_hash=digest.hexdigest())
    filename = prefix + 'likelihood.json'

    if resume:
        if os.path.exists(filename):
            with open(filename) as f:
                saved = json.load(f)

            if saved != signature:
                raise ValueError('Likelihood settings or measurements differ from the saved run; use a new NS_prefix')

            return

        if any(os.path.exists(prefix + suffix) for suffix in ('.txt', 'resume.dat', 'live.points', 'phys_live.points')):
            raise ValueError('Cannot verify the likelihood of existing chains without likelihood.json; use a new NS_prefix')

    _ns_save_json(filename, signature)


def _ns_legacy(runname,Variables,Measurement,Atmosphere,Spectroscopy,Scatter,Stellar,Surface,CIA,Layer,Telluric,
               NS_prefix='chains/',nemesisC=False):
    """
    Original ordinary sampler, including native output and checkpoint format.
    New prior, likelihood and transport settings are handled by coreretNS instead.
    """

    from archnemesis.ForwardModel_0 import ForwardModel_0
    from pymultinest.solve import solve
    from mpi4py import MPI

    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    prefix = os.fspath(NS_prefix)

    def prepare_output():
        if any(os.path.exists(filename) for filename in
               (prefix+'likelihood.json', prefix+'priors.json', prefix+'posterior.npz',
                os.path.join(prefix, 'repartitioning.json'))):
            raise ValueError('This NS_prefix contains new-framework output; supply its original NS settings or use a new NS_prefix')

        os.makedirs(prefix, exist_ok=True)
    _ns_rank_zero(comm, prepare_output)

    NestedSampling = NestedSampling_0(nemesisC=nemesisC)
    NestedSampling.ForwardModel = ForwardModel_0(runname=runname, Atmosphere=Atmosphere,Surface=Surface,
                                  Measurement=Measurement,Spectroscopy=Spectroscopy,Telluric=Telluric,
                                  Stellar=Stellar,Scatter=Scatter,CIA=CIA,Layer=Layer,Variables=Variables)
    NestedSampling.XA = Variables.XA
    NestedSampling.XA_ERR = np.sqrt(Variables.SA.diagonal())
    NestedSampling.Y = Measurement.Y
    NestedSampling.Y_ERR = np.sqrt(Measurement.SE.diagonal())
    NestedSampling.vars_to_vary = [i for i in range(len(NestedSampling.XA)) if NestedSampling.XA_ERR[i]>1e-5]
    NestedSampling.priors = [scipy.stats.norm(NestedSampling.XA[i], NestedSampling.XA_ERR[i]).ppf
                             for i in NestedSampling.vars_to_vary]
    NestedSampling.prefix = prefix
    NestedSampling.parameters = [str(i) for i in NestedSampling.vars_to_vary]
    NestedSampling.result = solve(LogLikelihood=NestedSampling.LogLikelihood,
                                  Prior=NestedSampling.Prior,
                                  n_dims=len(NestedSampling.parameters),
                                  outputfiles_basename=NestedSampling.prefix,
                                  verbose=True,
                                  n_live_points=NestedSampling.N_LIVE_POINTS,
                                  evidence_tolerance=0.5)

    if rank == 0:
        _lgr.info('')
        _lgr.info('Evidence: %(logZ).1f +- %(logZerr).1f' % NestedSampling.result)
        _lgr.info('')
        _lgr.info('Parameter values:')

        for name, col in zip(NestedSampling.parameters, NestedSampling.result['samples'].transpose()):
            _lgr.info('%15s : %.3f +- %.3f' % (name, col.mean(), col.std()))

    comm.barrier()

    return NestedSampling


@_ns_mpi_errors
def coreretNS(
        runname, Variables, Measurement, Atmosphere, Spectroscopy,
        Scatter, Stellar, Surface, CIA, Layer, Telluric,
        NS_prefix='chains/',
        nemesisC=False,
        Emulator=None,
        emulator_mean=None,
        emulator_cov=None,
        NS_options=None,
        guide_options=None,
        NS_normalise=None,
        NS_run_id=None,
        nemesisSO=False,
        nemesisdisc=False,
        nemesisPT=False,
        NS_full_covariance=None,
        NS_validate_state=None,
        NS_transport_options=None,
        NS_stage='both'
):
    """
        FUNCTION NAME : coreretNS()
        
        DESCRIPTION : 

            This subroutine runs Nested Sampling to fit an atmospheric model to a spectrum, and gives
            a good idea of the distribution of fitted parameters.
            Optional runname.nsp prior overrides are read only here, after .apr loading.
            See NestedSamplingPrior_0.read_nsp for the file format and coordinate rules.
            With no .nsp file or new options, use the original sampler and output
            format, allowing old chains to resume. NS_prefix and nemesisC retain
            their original meaning. Other observation-mode flags remain ignored
            on this legacy route. NS_options={} explicitly selects the new route.

        INPUTS :
       
            runname :: Name of the Nemesis run
            Variables :: Python class defining the parameterisations and state vector
            Measurement :: Python class defining the measurements 
            Atmosphere :: Python class defining the reference atmosphere
            Spectroscopy :: Python class defining the spectroscopic parameters of gaseous species
            Scatter :: Python class defining the parameters required for scattering calculations
            Stellar :: Python class defining the stellar spectrum
            Surface :: Python class defining the surface
            CIA :: Python class defining the Collision-Induced-Absorption cross-sections
            Layer :: Python class defining the layering scheme to be applied in the calculations
            Telluric :: Python class defining the parameters to calculate the Telluric absorption

        OPTIONAL INPUTS :

            Emulator :: Pretrained callable taking free Variables.XN entries in index order.
                        Returns a measurement vector or (max(NCONV),NGEOM) spectral array.
            emulator_mean :: Held-out mean of Emulator - full model, in measurement order.
                             Defaults to zero when Emulator is supplied.
            emulator_cov :: Held-out residual covariance, or vector of residual variances.
                            Required with Emulator; zero variances must be supplied explicitly.
            NS_options :: Dictionary of MultiNest settings for the full-model run.
            guide_options :: Dictionary of MultiNest settings for the guide run.
                             The guide runs serially on rank zero.
                             Resume is controlled by NS_options for both stages.
            NS_transport_options :: None retains the single Gaussian. To select a mixture,
                                    supply dict(type='gaussian_mixture', max_components=6, seed=0).
                                    Component count uses weighted five-fold validation.
            NS_stage :: 'both' (default) runs the guide and then the full model.
                        'guide' returns after fitting and saving the transport.
                        'full' loads a completed guide and runs only the full model.
                        Separate stages require NS_run_id. 'guide' requires Emulator;
                        'full' need not load it and reuses saved error arrays and guide/
                        transport settings when those arguments are omitted. Supply the
                        same scientific inputs and full-sampler settings. Resume in
                        NS_options controls restarting existing sampler output; a first
                        'full' invocation may use resume=False with an empty full folder.
            NS_normalise :: Include the Gaussian log-normalisation constant.
                            Defaults to True with Emulator, False otherwise.
            NS_full_covariance :: Use all of Measurement.SE instead of its diagonal.
                                  Defaults to True with Emulator, False otherwise,
                                  independently of NS_normalise. Emulator covariance is
                                  added in full to the selected measurement covariance.
            NS_run_id :: User identifier for the pretrained model, preprocessing and all
                         forward-model inputs. Supply on the first run to permit guided resume.
                         Change this identifier whenever these external inputs or the state
                         validator change. Also checked for ordinary runs when supplied.
            NS_validate_state :: Optional callable receiving the complete Variables.XN vector.
                                 Return False or raise InvalidAtmosphericState for zero likelihood.
                                 Applied to both guide and full likelihoods. Other exceptions abort.
                                 Requires NS_run_id to identify the validator for restarts.

        OUTPUTS :

            NestedSampling :: Python class containing information from the retrieval.
 
        CALLING SEQUENCE:
        
            NestedSampling = coreretNS(runname,Variables,Measurement,Atmosphere,Spectroscopy,Scatter,Stellar,Surface,CIA,Layer,Telluric)
 
        MODIFICATION HISTORY : Joe Penn (09/10/24)

    """
    
    
    from archnemesis.ForwardModel_0 import ForwardModel_0
    from archnemesis.NestedSampling_0 import NestedSampling_0
    from mpi4py import MPI

    start = time.time()
    # This function should be launched in parallel. We set up the MPI environment.
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    #size = comm.Get_size()

    legacy = NS_stage == 'both' and all(value is None for value in
        (Emulator, emulator_mean, emulator_cov, NS_options, guide_options, NS_normalise,
         NS_run_id, NS_full_covariance, NS_validate_state, NS_transport_options))
    if legacy and not _ns_rank_zero(comm, lambda: os.path.exists(os.fspath(runname)+'.nsp')):
        return _ns_legacy(runname,Variables,Measurement,Atmosphere,Spectroscopy,Scatter,Stellar,Surface,CIA,Layer,Telluric,
                          NS_prefix=NS_prefix,nemesisC=nemesisC)

    # Defining the NestedSampling class
    guided = Emulator is not None or NS_stage == 'full'
    if NS_stage not in ('both', 'guide', 'full'):
        raise ValueError("NS_stage must be 'both', 'guide', or 'full'")
    if NS_stage != 'both' and not guided:
        raise ValueError('Separate nested-sampling stages require Emulator')
    if NS_stage != 'both' and NS_run_id is None:
        raise ValueError('Supply NS_run_id to identify inputs across separate stages')
    if not guided and any(value is not None for value in (emulator_mean, emulator_cov, guide_options, NS_transport_options)):
        raise ValueError('Emulator inputs and guide settings require Emulator')
    if NS_run_id is not None and (not isinstance(NS_run_id, str) or not NS_run_id.strip()):
        raise ValueError('NS_run_id must be a non-empty string')

    if NS_validate_state is not None:
        if not callable(NS_validate_state):
            raise ValueError('NS_validate_state must be callable')
        if NS_run_id is None:
            raise ValueError('Supply NS_run_id to identify NS_validate_state for restarts')

    if NS_stage == 'full':
        saved = _ns_rank_zero(comm, lambda: _ns_read_guide_setup(NS_prefix))

        if guide_options is None:
            guide_options = saved['signature']['guide_options']

        if NS_transport_options is None:
            NS_transport_options = saved['signature'].get('transport_options')

        if emulator_mean is None or emulator_cov is None:
            mean, covariance = _ns_rank_zero(comm, lambda: _ns_read_guide_setup(NS_prefix, errors=True))

            if emulator_mean is None:
                emulator_mean = mean

            if emulator_cov is None:
                emulator_cov = covariance

    options = _ns_options(NS_options, guided=guided)
    if guide_options is not None and 'resume' in guide_options:
        raise ValueError('Set resume in NS_options, not guide_options')

    guide_settings = _ns_options(guide_options, guided=True, guide=True)
    guide_settings['resume'] = options['resume']

    NestedSampling = NestedSampling_0(N_LIVE_POINTS=options['n_live_points'], nemesisC=nemesisC)
    NestedSampling.TRANSPORT_OPTIONS = None

    if NS_transport_options is not None:
        if not isinstance(NS_transport_options, dict) or set(NS_transport_options)-{'type', 'max_components', 'seed'}:
            raise ValueError('Unknown NS_transport_options; expected type, max_components and seed')

        settings = dict(type='gaussian_mixture', max_components=6, seed=0)
        settings.update(NS_transport_options)
        if settings['type'] != 'gaussian_mixture':
            raise ValueError('NS_transport_options type must be gaussian_mixture')

        for name, minimum in (('max_components', 1), ('seed', 0)):
            if isinstance(settings[name], (bool, np.bool_)) or not isinstance(settings[name], (int, np.integer)) or settings[name] < minimum:
                raise ValueError('Invalid mixture setting: ' + name)

        NestedSampling.TRANSPORT_OPTIONS = settings

    NestedSampling.StateValidator = NS_validate_state
    NestedSampling.RUN_ID = NS_run_id
    NestedSampling.PriorDistribution = _ns_rank_zero(comm, lambda: _ns_read_priors(Variables, runname))
    NestedSampling.vars_to_vary = NestedSampling.PriorDistribution.vars_to_vary

    VariablesNS = copy.copy(Variables)
    VariablesNS.XN = NestedSampling.PriorDistribution.XN.copy()
    
    NestedSampling.ForwardModel = ForwardModel_0(runname=runname, Atmosphere=Atmosphere,Surface=Surface,
                                  Measurement=Measurement,Spectroscopy=Spectroscopy,Telluric=Telluric,
                                  Stellar=Stellar,Scatter=Scatter,CIA=CIA,Layer=Layer,Variables=VariablesNS)

    NestedSampling.XA = Variables.XA
    NestedSampling.XA_ERR = np.sqrt(Variables.SA.diagonal())
    NestedSampling.Y = Measurement.Y
    NestedSampling.Y_ERR = np.sqrt(Measurement.SE.diagonal())

    NestedSampling.NORMALISE = guided if NS_normalise is None else bool(NS_normalise)
    NestedSampling.FULL_COVARIANCE = guided if NS_full_covariance is None else bool(NS_full_covariance)
    NestedSampling.ForwardModelFunction = NestedSampling.ForwardModel.select_nemesis_fm(
        nemesisSO=nemesisSO, nemesisdisc=nemesisdisc, nemesisPT=nemesisPT, nemesisC=nemesisC)
    if not np.all(np.isfinite(NestedSampling.Y)):
        raise ValueError('Measurements must be finite')

    scientific_covariance = Measurement.SE if NestedSampling.FULL_COVARIANCE else np.diag(Measurement.SE.diagonal())
    NestedSampling.LIKELIHOOD_SETUP = NestedSampling.gaussian_setup(scientific_covariance)

    # Making the retrieval folder
    NestedSampling.prefix = os.fspath(NS_prefix)
    NestedSampling.STAGE = 'full'
    _ns_rank_zero(comm, lambda: os.makedirs(NestedSampling.prefix, exist_ok=True))
    NestedSampling.parameters = [str(i) for i in NestedSampling.vars_to_vary]
    NestedSampling.N_OUTPUT_PARAMETERS = 2*len(NestedSampling.parameters) + 2

    if not guided:
        _ns_rank_zero(comm, lambda: _ns_prepare_likelihood(
            NestedSampling.prefix, NestedSampling, scientific_covariance, options['resume']))
        _ns_rank_zero(comm, lambda: _ns_prepare_priors(
            NestedSampling.prefix, NestedSampling.PriorDistribution, options['resume']))

    if guided:
        if (Emulator is not None or NS_stage != 'full') and not callable(Emulator):
            raise ValueError('Emulator must be a pretrained prediction callable')
        if emulator_cov is None:
            raise ValueError('Supply held-out emulator_cov when using an Emulator')

        NestedSampling.Emulator = Emulator
        NestedSampling.EMULATOR_MEAN = (
            np.zeros_like(NestedSampling.Y) if emulator_mean is None
            else np.array(emulator_mean, dtype=float, copy=True)
        )
        if NestedSampling.EMULATOR_MEAN.shape != NestedSampling.Y.shape or not np.all(np.isfinite(NestedSampling.EMULATOR_MEAN)):
            raise ValueError('emulator_mean must have one finite value per measurement')

        covariance = np.array(emulator_cov, dtype=float, copy=True)

        if covariance.ndim == 1:
            if covariance.shape != NestedSampling.Y.shape or np.any(covariance < 0.0):
                raise ValueError('emulator_cov vector must contain NY non-negative variances')

            covariance = np.diag(covariance)

        if covariance.shape != Measurement.SE.shape or not np.all(np.isfinite(covariance)):
            raise ValueError('emulator_cov must be a finite NY by NY covariance matrix')
        if not np.allclose(covariance, covariance.T, rtol=1.0e-10, atol=0.0):
            raise ValueError('emulator_cov must be symmetric')

        eigenvalues = (
            covariance.diagonal() if np.array_equal(covariance, np.diag(covariance.diagonal()))
            else np.linalg.eigvalsh(covariance)
        )
        if eigenvalues.min() < -1.0e-10 * max(np.abs(eigenvalues).max(), np.finfo(float).tiny):
            raise ValueError('emulator_cov must be positive semidefinite')

        NestedSampling.EMULATOR_COV = covariance
        NestedSampling.GUIDE_SETUP = NestedSampling.gaussian_setup(scientific_covariance + covariance)

        # Only rank zero evaluates the serial emulator guide.
        if NS_stage != 'full':
            error = _ns_rank_zero(comm, lambda: _ns_check_emulator(NestedSampling))
            if error is not None:
                raise ValueError('Emulator validation failed: ' + error)

        signature = _ns_signature(NestedSampling, Variables, Measurement, runname, NS_run_id,
                                  options, guide_settings, (nemesisSO, nemesisdisc, nemesisPT, nemesisC))
        signatures = comm.allgather(signature)
        if any(value != signature for value in signatures):
            raise ValueError('Guided run settings must be identical on every MPI rank')

        NestedSampling.GUIDE_PREFIX = os.path.join(NS_prefix, 'guide', '')
        coefficients = _ns_rank_zero(comm, lambda: _ns_prepare_transport(NS_prefix, signature, options['resume'], NS_stage))

        if coefficients is None:
            _ns_rank_zero(comm, lambda: _ns_write_guide_setup(NestedSampling, NS_prefix))
            NestedSampling.guide_result = _ns_run(NestedSampling, comm, NestedSampling.GUIDE_PREFIX,
                                                 guide_settings, guide=True)
            coefficients = _ns_rank_zero(comm, lambda: _ns_fit_transport(NestedSampling, NS_prefix, signature))
            NestedSampling.ELAPSED_SECONDS = time.time() - start
            _ns_rank_zero(comm, lambda: _ns_write_result(NestedSampling, Variables,
                         prefix=NestedSampling.GUIDE_PREFIX, result=NestedSampling.guide_result, stage='guide'))
        else:
            NestedSampling.guide_result = _ns_rank_zero(comm, lambda: _ns_read_result(NestedSampling, NestedSampling.GUIDE_PREFIX))

        NestedSampling.Transport = NestedSamplingTransport_0.from_coefficients(coefficients)
        saved = _ns_rank_zero(comm, lambda: _ns_read_guide_setup(NS_prefix))
        NestedSampling.TRANSPORT_SELECTION = saved.get('selection')
        NestedSampling.prefix = os.path.join(NS_prefix, 'full', '')

        if NS_stage == 'guide':
            NestedSampling.STAGE = 'guide'
            NestedSampling.prefix = NestedSampling.GUIDE_PREFIX
            NestedSampling.result = NestedSampling.guide_result

    # run MultiNest
    if NS_stage != 'guide':
        NestedSampling.result = _ns_run(NestedSampling, comm, NestedSampling.prefix, options)

    NestedSampling.N_FULL_CALLS = comm.allreduce(NestedSampling.N_FULL_CALLS, op=MPI.SUM)
    NestedSampling.N_GUIDE_CALLS = comm.allreduce(NestedSampling.N_GUIDE_CALLS, op=MPI.SUM)
    NestedSampling.ELAPSED_SECONDS = time.time() - start
    _ns_rank_zero(comm, lambda: _ns_write_result(NestedSampling, Variables))

    #Print parameters
    if rank == 0:
        _lgr.info('')
        _lgr.info('Evidence: %(logZ).1f +- %(logZerr).1f' % NestedSampling.result)
        _lgr.info('')
        _lgr.info('Parameter values:')

        for name, col in zip(NestedSampling.parameters, NestedSampling.result['samples'].transpose()):
            _lgr.info('%15s : %.3f +- %.3f' % (name, col.mean(), col.std()))

    comm.barrier()

    return NestedSampling
