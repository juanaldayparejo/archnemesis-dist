


import dataclasses as dc
from typing import TYPE_CHECKING

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt

import archnemesis as ans

from .value import Value

if TYPE_CHECKING:
	from ..Retrieval import Retrieval

import archnemesis.cfg.logs as logging
_lgr = logging.getLogger(__name__)
_lgr.setLevel(logging.INFO)



@dc.dataclass(slots=True)
class OptimalEstimationPlotter:
	retrieval : "Retrieval"
	
	def can_plot(self):
		return self.retrieval.RetrievalEngine is not None and isinstance(self.retrieval.RetrievalEngine, ans.OptimalEstimation_0)
	
	def plot(self) -> tuple[str, mpl.figure.Figure, tuple[np.ndarray[mpl.axes.Axes]]]:
		return 'optimal_estimation', *self.do_plot(self.retrieval.RetrievalEngine)

	def do_plot(self, RetrievalEngine, f=None) -> tuple[mpl.figure.Figure, tuple[np.ndarray[mpl.axes.Axes]]]:
		from mpl_toolkits.axes_grid1 import make_axes_locatable

		# fig info
		fig_spec = dict(
			figsize=(12,12)
		)
		
		# subfig info
		subfig_spec = dict(
			nrows=1, 
			ncols=1, 
			squeeze=False
		)
		
		# subplot mosaic info
		subplot_layout = [
			['spectra_values', 'spectra_values', 'spectra_linear'],
			['spectra_residual', 'spectra_residual', 'spectra_log'],
			['jacobian', 'jacobian', 'jacobian'],
			['covariance', 'correlation', None],
		]
		
		# gridspec info
		gridspec_kw = dict(
			wspace=None,
			hspace=0.3
		)
		
		
		f = plt.figure(**fig_spec) if f is None else f
		subfigs = f.subfigures(**subfig_spec).flatten()
		axes = tuple(subfig.subplot_mosaic(subplot_layout, gridspec_kw=gridspec_kw) for subfig in subfigs)

		for subfig, sf_axes in zip(subfigs, axes):
			if hasattr(RetrievalEngine, 'PHI'):
				subfig.suptitle(f'Optimal Estimation\nphi (cost function) : {RetrievalEngine.PHI}\nchisq (goodness of fit) : {RetrievalEngine.CHISQ}')
			else:
				subfig.suptitle('Optimal Estimation')
			
			ax_iter = iter(sf_axes.values())
			
			with Value(next(ax_iter)) as ax:
				
				_title = 'Measured and modelled spectra'
				if RetrievalEngine.Y is not None:
					ax.plot(
						range(RetrievalEngine.NY),
						RetrievalEngine.Y,
						c='tab:blue',
						label='Measured spectra',
						alpha=0.6
					)
					
					if RetrievalEngine.SE is not None:
						y_err = np.sqrt(np.diag(RetrievalEngine.SE))
						ax.fill_between(
							range(RetrievalEngine.NY),
							RetrievalEngine.Y-y_err,
							RetrievalEngine.Y+y_err,
							color='tab:blue',
							label='Measured spectra error (1 sigma)',
							alpha=0.3
						)
					
					
					
				else:
					_title += '\nOriginal value not present, cannot plot measured spectra'
				
				if RetrievalEngine.YN is not None:
					ax.plot(
						range(RetrievalEngine.NY),
						RetrievalEngine.YN,
						c='tab:orange',
						label='Modelled spectra',
						alpha=0.6
					)
				else:
					_title += '\nRetrieved value not present, cannot plot modelled spectra'
				
				ax.set_title(_title)
				ax.set_xlabel('Measurement vector element #')
				ax.set_ylabel('Radiance')
				ax.grid()
				ax.legend()
			
			with Value(next(ax_iter)) as ax:
				_title = 'Spectra residual'
				if RetrievalEngine.Y is not None and RetrievalEngine.YN is not None:
					ax.plot(
						range(RetrievalEngine.NY),
						RetrievalEngine.Y-RetrievalEngine.YN,
						c='tab:red',
						alpha=0.6
					)
					
					if RetrievalEngine.SE is not None:
						y_err = np.sqrt(np.diag(RetrievalEngine.SE))
						ax.fill_between(
							range(RetrievalEngine.NY),
							RetrievalEngine.Y-y_err - RetrievalEngine.YN,
							RetrievalEngine.Y+y_err - RetrievalEngine.YN,
							color='tab:red',
							label='Spectra residual error (1 sigma)',
							alpha=0.3
						)
					
				else:
					_title += '\nEither original or retrieved value not present, cannot plot residual'
				
				ax.set_title(_title)
				ax.set_xlabel('Measurement vector element #')
				ax.set_ylabel('Residual Radiance')
				ax.legend()
				ax.grid()
			
			with Value(next(ax_iter)) as ax:
				_title = 'apriori and retrieved state vector'
				
				if RetrievalEngine.XA is not None:
					ax.plot(
						range(RetrievalEngine.NX),
						RetrievalEngine.XA,
						c='tab:blue',
						label='Apriori state vector',
						alpha=0.6
					)
					
					if RetrievalEngine.SA is not None:
						sv_err = np.sqrt(np.diag(RetrievalEngine.SA))
						ax.fill_between(
							range(RetrievalEngine.NX),
							RetrievalEngine.XA-sv_err,
							RetrievalEngine.XA+sv_err,
							color='tab:blue',
							label='apriori error (1 sigma)',
							alpha=0.3
						)
					
				else:
					_title += '\nApriori state vector not present'
				
				if RetrievalEngine.XN is not None:
					ax.plot(
						range(RetrievalEngine.NX),
						RetrievalEngine.XN,
						c='tab:orange',
						label='Retrieved state vector',
						alpha=0.6
					)
				else:
					_title += '\nRetrieved state vector not present'
				
				ax.set_title(_title)
				ax.set_xlabel('State vector element #')
				ax.set_ylabel('State vector element value')
				ax.grid()
				ax.legend()
			
			with Value(next(ax_iter)) as ax:
				_title = 'state vector residual'
				if RetrievalEngine.XA is not None and RetrievalEngine.XN is not None:
					ax.plot(
						range(RetrievalEngine.NX),
						RetrievalEngine.XA-RetrievalEngine.XN,
						c='tab:red',
						alpha=0.6
					)
					
					if RetrievalEngine.SA is not None:
						sv_err = np.sqrt(np.diag(RetrievalEngine.SA))
						ax.fill_between(
							range(RetrievalEngine.NX),
							RetrievalEngine.XA-sv_err -RetrievalEngine.XN,
							RetrievalEngine.XA+sv_err -RetrievalEngine.XN,
							color='tab:red',
							label='state vector residual error (1 sigma)',
							alpha=0.3
						)
					
				else:
					_title += '\nEither apriori or retrieved state vector is not present, cannot plot residual'
					
				ax.set_title(_title)
				ax.set_xlabel('State vector element #')
				ax.set_ylabel('Residual value of state vector element')
				ax.legend()
				ax.grid()
			
			with Value(next(ax_iter)) as ax:
				_lgr.debug(f'{RetrievalEngine.KK=}')
			
				if (RetrievalEngine.KK is None
						or RetrievalEngine.KK.size == 0
						or RetrievalEngine.KK.ndim == 0
						or RetrievalEngine.KK[0] is None
					):
					_lgr.warning('Optimal Estimation Jacobian Matrix has not been set or is invalid')
					ax.remove()
					break
				
				ax.set_title('Jacobian Matrix')
				# Center limits around zero
				vmin = np.nanmin(RetrievalEngine.KK)
				vmax = np.nanmax(RetrievalEngine.KK)
				centered_limit = max(abs(vmin),abs(vmax))
				
				im = ax.imshow(
					np.transpose(RetrievalEngine.KK),
					aspect='auto',
					origin='lower',
					cmap='bwr', 
					vmin=-centered_limit, 
					vmax=centered_limit,
					interpolation='nearest'
				)
				divider = make_axes_locatable(ax)
				cax = divider.append_axes("right", size="5%", pad=0.05)
				cbar = plt.colorbar(im, cax=cax)
				cbar.set_label('Gradients (dR/dx)')
				ax.set_ylabel('state vector element #')
				ax.set_xlabel('measurement vector element #')
				ax.grid()
			
			
			with Value(next(ax_iter)) as ax:
				_lgr.debug(f'{RetrievalEngine.KK=}')
			
				if (RetrievalEngine.KK is None
						or RetrievalEngine.ST.size == 0
						or RetrievalEngine.ST.ndim == 0
						or RetrievalEngine.ST[0] is None
					):
					_lgr.warning('Optimal Estimation Covariance Matrix has not been set or is invalid')
					ax.remove()
					break
				
				
				ax.set_title('Covariance Matrix')
				# Center limits around zero
				vmin = np.nanmin(RetrievalEngine.ST)
				vmax = np.nanmax(RetrievalEngine.ST)
				centered_limit = max(abs(vmin),abs(vmax))
				
				im = ax.imshow(
					np.transpose(RetrievalEngine.ST),
					aspect=1,
					origin='lower',
					cmap='bwr', 
					vmin=-centered_limit, 
					vmax=centered_limit,
					interpolation='nearest'
				)
				divider = make_axes_locatable(ax)
				cax = divider.append_axes("right", size="5%", pad=0.05)
				cbar = plt.colorbar(im, cax=cax)
				cbar.set_label('Covariance Value')
				ax.set_ylabel('state vector element #')
				ax.set_xlabel('state vector element #')
				ax.grid()

			with Value(next(ax_iter)) as ax:
				_lgr.debug(f'{RetrievalEngine.KK=}')
			
				if (RetrievalEngine.KK is None
						or RetrievalEngine.ST.size == 0
						or RetrievalEngine.ST.ndim == 0
						or RetrievalEngine.ST[0] is None
					):
					_lgr.warning('Optimal Estimation Covariahce Matrix has not been set or is invalid')
					ax.remove()
					break
				
				ax.set_title('Correlation Matrix')
				# Center limits around zero
				vmin = np.nanmin(RetrievalEngine.ST)
				vmax = np.nanmax(RetrievalEngine.ST)
				centered_limit = max(abs(vmin),abs(vmax))
				
				sigmas =  np.sqrt(np.diagonal(RetrievalEngine.ST))
				correlation_matrix = np.transpose(RetrievalEngine.ST / sigmas) / sigmas
				correlation_matrix += np.diag(np.full((correlation_matrix.shape[0],), fill_value=np.nan)) # set all diagonals to NAN
				im = ax.imshow(
					correlation_matrix,
					aspect=1,
					origin='lower',
					cmap='bwr', 
					vmin=-centered_limit, 
					vmax=centered_limit,
					interpolation='nearest',
				)
				divider = make_axes_locatable(ax)
				cax = divider.append_axes("right", size="5%", pad=0.05)
				cbar = plt.colorbar(im, cax=cax)
				cbar.set_label('Correlation Value\n(diagonals removed for clarity)')
				ax.set_ylabel('state vector element #')
				ax.set_xlabel('state vector element #')
				ax.grid()

			# clean up unused axes
			for ax in ax_iter:
				ax.remove()
		
		return f, axes
	