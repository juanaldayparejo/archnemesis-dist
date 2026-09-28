


import dataclasses as dc
from typing import TYPE_CHECKING

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt

from .value import Value

if TYPE_CHECKING:
	from ..Retrieval import Retrieval

import archnemesis.cfg.logs as logging
_lgr = logging.getLogger(__name__)
_lgr.setLevel(logging.INFO)



@dc.dataclass(slots=True)
class CIAPlotter:
	retrieval : "Retrieval"
	
	def can_plot(self):
		return self.retrieval.CIA is not None
	
	def plot(self) -> tuple[str, mpl.figure.Figure, tuple[np.ndarray[mpl.axes.Axes]]]:
		return 'cia', *self.do_plot(self.retrieval.CIA)

	def do_plot(self, CIA, f = None) -> tuple[mpl.figure.Figure, tuple[np.ndarray[mpl.axes.Axes]]]:
		from archnemesis.Data.gas_data import gas_info

		# fig info
		fig_spec = dict(
			figsize=(12,8)
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
			ncols=2, 
			squeeze=False
		)
		
		
		f = plt.figure(**fig_spec) if f is None else f
		subfigs = f.subfigures(**subfig_spec).flatten()
		axes = tuple(subfig.subplots(**subplot_spec).flatten() for subfig in subfigs)
		
		for subfig, sf_axes in zip(subfigs, axes):
			subfig.suptitle('Collision Induced Absorption')
			ax_iter = iter(sf_axes)
			
			with Value(next(ax_iter)) as ax:
				#labels = ['H$_2$-H$_2$ w equilibrium ortho/para-H$_2$','He-H$_2$ w equilibrium ortho/para-H$_2$','H$_2$-H$_2$ w normal ortho/para-H$_2$','He-H$_2$ w normal ortho/para-H$_2$','H$_2$-N$_2$','N$_2$-CH$_4$','N$_2$-N$_2$','CH$_4$-CH$_4$','H$_2$-CH$_4$']
				for i in range(CIA.NPAIR):
					gasname1 = gas_info[str(CIA.IPAIRG1[i])]['name']
					gasname2 = gas_info[str(CIA.IPAIRG2[i])]['name']

					label = gasname1+'-'+gasname2
					if CIA.INORMALT[i]==1:
						label = label + " ('normal')"

					iTEMP = np.argmin(np.abs(CIA.TEMP-296.))
					
					for j in range(max(CIA.NPARA,1)):
						label += ' ortho' if j==0 else ' para'
					
						ax.plot(CIA.WAVEN, CIA.K_CIA[i,j,iTEMP,:], label=label, alpha=0.6)

				ax.legend()
				#ax.set_facecolor('lightgray')
				ax.set_xlabel('Wavenumber (cm$^{-1}$')
				ax.set_ylabel('CIA cross section (cm$^{5}$ molec$^{-2}$)')
				ax.grid()
			
			with Value(next(ax_iter)) as ax:
				#labels = ['H$_2$-H$_2$ w equilibrium ortho/para-H$_2$','He-H$_2$ w equilibrium ortho/para-H$_2$','H$_2$-H$_2$ w normal ortho/para-H$_2$','He-H$_2$ w normal ortho/para-H$_2$','H$_2$-N$_2$','N$_2$-CH$_4$','N$_2$-N$_2$','CH$_4$-CH$_4$','H$_2$-CH$_4$']
				for i in range(CIA.NPAIR):
					gasname1 = gas_info[str(CIA.IPAIRG1[i])]['name']
					gasname2 = gas_info[str(CIA.IPAIRG2[i])]['name']

					label = gasname1+'-'+gasname2
					if CIA.INORMALT[i]==1:
						label = label + " ('normal')"

					iTEMP = np.argmin(np.abs(CIA.TEMP-296.))
					
					for j in range(max(CIA.NPARA,1)):
						label += ' ortho' if j==0 else ' para'
					
						ax.plot(CIA.WAVEN, CIA.K_CIA[i,j,iTEMP,:], label=label, alpha=0.6)

				#ax.legend()
				#ax.set_facecolor('lightgray')
				ax.set_xlabel('Wavenumber (cm$^{-1}$')
				ax.set_ylabel('log[CIA cross section] (log[cm$^{5}$ molec$^{-2}$])')
				ax.grid()
				ax.set_yscale('log')
				
				
		
		return f, axes
	