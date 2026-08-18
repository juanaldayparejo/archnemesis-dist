


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
class LayerPlotter:
	retrieval : "Retrieval"
	
	def can_plot(self):
		return self.retrieval.Layer is not None
	
	def plot(self) -> tuple[str, mpl.figure.Figure, tuple[np.ndarray[mpl.axes.Axes]]]:
		if self.retrieval.Layer is None:
			self.retrieval.calculate_layering()
		
		return 'layer', *self.do_plot(self.retrieval.Layer, self.retrieval.Atmosphere.ID, self.retrieval.Atmosphere.ISO)

	def do_plot(self,Layer, gas_ids, gas_iso_ids, f=None) -> tuple[mpl.figure.Figure, tuple[np.ndarray[mpl.axes.Axes]]]:
	
		# fig info
		fig_spec = dict(
			figsize=(12,4)
		)
		
		# subfig info
		subfig_spec = dict(
			nrows=1, 
			ncols=1, 
			squeeze=False
		)
		
		# subplot info
		subplot_spec = dict(
			nrows=1, 
			ncols=5, # last column will be for the legend
			squeeze=False,
			sharey=True,
		)
		
		# style info
		hline_style = dict(
			linestyle = '--',
			linewidth=0.5,
			alpha=0.6,
			color='black',
		)
		
		plotline_style = dict(
			marker='x',
			markersize=4,
			linewidth=1,
			alpha=0.6,
		)
		
		
		f = plt.figure(**fig_spec) if f is None else f
		subfigs = f.subfigures(**subfig_spec).flatten()
		axes = tuple(subfig.subplots(**subplot_spec).flatten() for subfig in subfigs)
	
		
		for subfig, sf_axes in zip(subfigs, axes):
			if Layer.BASEH is None:
				_lgr.warning('Cannot create plot of Layer as they have not been calculated yet')
				subfig.suptitle('Layers have not been calculated yet')
				for ax in sf_axes:
					ax.remove()
				return f, axes
			else:
				subfig.suptitle(f'Layer Summary number_of_layers = {Layer.NLAY}')
				for ax in sf_axes[1:]:
					ax.sharey(sf_axes[0])
				
			ax_iter = iter(sf_axes)
		
			#Pressure
			with Value(next(ax_iter)) as ax:
				for i in range(Layer.NLAY):
					ax.axhline(Layer.BASEH[i]/1.0e3, **hline_style)
				ax.plot(Layer.PRESS/1.0e5,Layer.HEIGHT/1.0e3, label = 'pressure', **plotline_style)
				ax.set_xscale('log')
				ax.set_xlabel('Effective pressure (bar)')
				ax.set_ylabel('Altitude (km)')

			#Temperature
			with Value(next(ax_iter)) as ax:
				for i in range(Layer.NLAY):
					ax.axhline(Layer.BASEH[i]/1.0e3, **hline_style)
				ax.plot(Layer.TEMP,Layer.HEIGHT/1.0e3, label='temp', **plotline_style)
				ax.set_xlabel('Effective temperature (K)')
				#ax.set_ylabel('Altitude (km)')

			
			#Gaseous column density
			with Value(next(ax_iter)) as ax:
				for i in range(Layer.NLAY):
					ax.axhline(Layer.BASEH[i]/1.0e3, **hline_style)
				for i in range(Layer.PP.shape[1]):
					ax.plot(Layer.AMOUNT[:,i],Layer.HEIGHT/1.0e3, label=f'{ans.Data.gas_info[str(gas_ids[i])]["name"]} isotope {gas_iso_ids[i]}', **plotline_style)
				ax.set_xscale('log')
				ax.set_xlabel('Gaseous column density (m$^{-2}$)')
				#ax.set_ylabel('Altitude (km)')

			#Dust column density
			with Value(next(ax_iter)) as ax:
				for i in range(Layer.NLAY):
					ax.axhline(Layer.BASEH[i]/1.0e3, **hline_style)
				for i in range(Layer.CONT.shape[1]):
					ax.plot(Layer.CONT[:,i],Layer.HEIGHT/1.0e3, label=f'aerosol_species_index {i}', **plotline_style)
				ax.set_xscale('log')
				ax.set_xlabel('Aerosol column density (m$^{-2}$)')
				#ax.set_ylabel('Altitude (km)')
			
			# Remove the last axes as we want to put the legend there
			with Value(next(ax_iter)) as ax:
				subfig.legend(loc = 'upper left', bbox_to_anchor=(ax.get_position().xmin, ax.get_position().ymax))
				ax.remove()
				
		
		return f, axes
	