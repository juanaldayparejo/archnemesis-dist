import archnemesis as ans
from archnemesis import *
import scipy
try:
    import pymultinest
    from pymultinest.solve import solve
except ImportError:
    _lgr.critical('PyMultiNest is not installed. Please download this before attempting to run retrievals with nested sampling. Instructions on installation can be found here: http://johannesbuchner.github.io/PyMultiNest/install.html')
    pymultinest = None
    solve = None
import os
import corner
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

import logging
_lgr = logging.getLogger(__name__)
_lgr.setLevel(logging.DEBUG)


class NestedSampling_0:
    
    def __init__(self, N_LIVE_POINTS=400):
        
        """
        Inputs
        ------
        @param N_LIVE_POINTS: int,
            Number of live points in retrieval 

        Methods
        -------
        NestedSampling.reduced_chi_squared()
        NestedSampling.LogLikelihood()
        NestedSampling.Prior()
        NestedSampling.make_plots()
        """
        
        
        self.N_LIVE_POINTS = N_LIVE_POINTS
        if pymultinest is None:
            raise ImportError("pymultinest was not found. To use the NestedSampling class, you must install it: \n https://johannesbuchner.github.io/PyMultiNest/install.html") 
            
            
    def chi_squared(self, a,b,err):
        """
        Calculate chi^2/n statistic.
        """        
        return np.sum(((a - b.T.flatten())**2)/(err**2))

    def LogLikelihood(self,cube):
        """
        Compute likelihood - run a forward model and compare to spectrum.
        """   
        
        self.ForwardModel.Variables.XN[self.vars_to_vary] = cube 
        
        original_stdout = sys.stdout  
        try:
            sys.stdout = open(os.devnull, 'w')  # Redirect stdout
            YN = self.ForwardModel.nemesisfm()
        finally:
            sys.stdout.close()  # Close the devnull
            sys.stdout = original_stdout  # Restore the original stdout
        
        return -self.chi_squared(self.Y,YN,self.Y_ERR)/2
    
    def Prior(self, cube):
        """
        Map unit cube to prior distributions.
        """  
        
        cube1 = cube.copy()
        for i in range(len(self.vars_to_vary)):
              cube1[i] = self.priors[i](cube1[i])

        return cube1
    
    def make_plots(self):
        """
        Cornerplot of results with analytical prior.
        """

        prior_means = self.XA
        prior_stds = self.XA_ERR

        # Initialize the analyzer
        a = pymultinest.Analyzer(n_params=len(self.parameters), outputfiles_basename=self.prefix)
        s = a.get_stats()

        _lgr.info('Creating marginal plot ...')

        # Extract data and weights
        data_array = a.get_data()
        weights = data_array[:, 0]
        data = data_array[:, 2:]

        # Apply weight mask (optional, depending on your data)
        mask = weights > 1e-4
        data_masked = data[mask, :]
        weights_masked = weights[mask]

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
            y = (
                (1 / (np.sqrt(2 * np.pi) * prior_stds[self.vars_to_vary[i]]))
                * np.exp(-0.5 * ((x - prior_means[self.vars_to_vary[i]]) / prior_stds[self.vars_to_vary[i]])**2)
            )
            y *= np.max(np.histogram(data_masked[:, i], bins=50, density=True)[0])
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
        analyzer = pymultinest.Analyzer(n_params=len(self.parameters), outputfiles_basename=self.prefix)
        data_array = analyzer.get_data()
        weights = data_array[:, 0]
        samples = data_array[:, 2:]

        # Mask low-weight samples (optional)
        mask = weights > 1e-4
        samples_masked = samples[mask, :]
        weights_masked = weights[mask]

        # Load the covariance matrix for optimal estimation
        full_covariance_matrix = read_cov(self.ForwardModel.runname)[9]

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
        
def coreretNS(runname,Variables,Measurement,Atmosphere,Spectroscopy,Scatter,Stellar,Surface,CIA,Layer,Telluric,NS_prefix='chains/'):
    """
        FUNCTION NAME : coreretNS()
        
        DESCRIPTION : 

            This subroutine runs Nested Sampling to fit an atmospheric model to a spectrum, and gives
            a good idea of the distribution of fitted parameters.

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

        OUTPUTS :

            NestedSampling :: Python class containing information from the retrieval.
 
        CALLING SEQUENCE:
        
            NestedSampling = coreretNS(runname,Variables,Measurement,Atmosphere,Spectroscopy,Scatter,Stellar,Surface,CIA,Layer,Telluric)
 
        MODIFICATION HISTORY : Joe Penn (09/10/24)

    """
    
    
    from archnemesis.ForwardModel_0 import ForwardModel_0
    from archnemesis.NestedSampling_0 import NestedSampling_0
    from mpi4py import MPI

    # This function should be launched in parallel. We set up the MPI environment.
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()

    # Defining the NestedSampling class
    NestedSampling = NestedSampling_0()
    
    NestedSampling.ForwardModel = ForwardModel_0(runname=runname, Atmosphere=Atmosphere,Surface=Surface,
                                  Measurement=Measurement,Spectroscopy=Spectroscopy,Telluric=Telluric,
                                  Stellar=Stellar,Scatter=Scatter,CIA=CIA,Layer=Layer,Variables=Variables)

    NestedSampling.XA = Variables.XA
    NestedSampling.XA_ERR = np.sqrt(Variables.SA.diagonal())
    NestedSampling.Y = Measurement.Y
    NestedSampling.Y_ERR = np.sqrt(Measurement.SE.diagonal())

    # Setting up prior distributions - right now, this just sets up log-gaussians with a standard deviation
    # equal to what is specified in the .apr file. There is support for log-uniform distributions for dist_code=1.
    
    NestedSampling.vars_to_vary = [i for i in range(len(NestedSampling.XA)) if NestedSampling.XA_ERR[i]>1e-5]

    NestedSampling.priors = []

    for i in NestedSampling.vars_to_vary:
        dist_code = 0                              ### PLACEHOLDER - need to add custom distributions!
        if dist_code == 0:
            NestedSampling.priors.append(scipy.stats.norm(NestedSampling.XA[i], NestedSampling.XA_ERR[i]).ppf)
        elif dist_code == 1:
            NestedSampling.priors.append(lambda x, i=i: x * (NestedSampling.XA[i] + NestedSampling.XA_ERR[i] - \
                                                             NestedSampling.XA[i] + 5*NestedSampling.XA_ERR[i]) + \
                                                             NestedSampling.XA[i] - 5*NestedSampling.XA_ERR[i])
        else:  
            _lgr.info(f'DISTRIBUTION ID NOT DEFINED!')

    # Making the retrieval folder
    NestedSampling.prefix = NS_prefix
    if rank == 0:
        if not os.path.exists(NestedSampling.prefix):
            os.makedirs(NestedSampling.prefix)
    comm.barrier()
    
    NestedSampling.parameters = [str(i) for i in NestedSampling.vars_to_vary]
    # run MultiNest
    NestedSampling.result = solve(LogLikelihood=NestedSampling.LogLikelihood, 
                                  Prior=NestedSampling.Prior, 
                                  n_dims=len(NestedSampling.parameters), 
                                  outputfiles_basename=NestedSampling.prefix, 
                                  verbose=True, 
                                  n_live_points = NestedSampling.N_LIVE_POINTS,
                                  evidence_tolerance = 0.5)

    #Print parameters
    if rank == 0:
        _lgr.info()
        _lgr.info('Evidence: %(logZ).1f +- %(logZerr).1f' % NestedSampling.result)
        _lgr.info()
        _lgr.info('Parameter values:')
        for name, col in zip(NestedSampling.parameters, NestedSampling.result['samples'].transpose()):
            _lgr.info('%15s : %.3f +- %.3f' % (name, col.mean(), col.std()))
    comm.barrier()
    return NestedSampling