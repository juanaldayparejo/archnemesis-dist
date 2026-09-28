
import dataclasses as dc
from typing import TYPE_CHECKING

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt

import archnemesis as ans
from archnemesis.Models.ModelBase import ModelBase

from .value import Value

if TYPE_CHECKING:
	from ..Retrieval import Retrieval



@dc.dataclass(slots=True)
class AtmospherePlotter:
	retrieval : "Retrieval"
	
	def can_plot(self):
		return self.retrieval.Atmosphere is not None
	
	def plot(self) -> tuple[str, mpl.figure.Figure, tuple[np.ndarray[mpl.axes.Axes]]]:
		return (
			'atmosphere', 
			*self.do_plot(
				self.retrieval.Atmosphere, 
				self.retrieval.get_all_gas_profile_models(), 
				self.retrieval.get_all_aerosol_profile_models()
			)
		)
	
	@staticmethod
	def _array_with_nth_axis(a : np.number | np.ndarray, n : int, new_axis : int = -1):
		"""
		Takes an array and adds a location axis to it if it does not already have one
		"""
		if isinstance(a, (np.number,float,int)):
			return np.array(a).reshape(tuple(1 for _ in range(n)))
		
		m = a.ndim
		s = tuple(1 for _ in range(n-m))
		if m < n:
			if new_axis == -1:
				return a.reshape((*a.shape, *s))
			elif new_axis == 0:
				return a.reshape((*s, *a.shape))
			else:
				raise ValueError(f'Unknown value to {new_axis=}. Should be either 0 (prepend new axes) or -1 (append new axes)')
	
	
	def do_plot(
			self,
			Atmosphere, 
			gas_vmr_models : list[None|ModelBase,...], 
			aerosol_density_models : list[None|ModelBase,...], 
			f = None,
	) -> tuple[mpl.figure.Figure, tuple[np.ndarray[mpl.axes.Axes]]]:
		# testing plotting with all information from profiles

		atm = Atmosphere

		# Get values we want to plot, make the single and multiple location versions the same shape
		latitude_loc = self._array_with_nth_axis(atm.LATITUDE, 1)
		longitude_loc = self._array_with_nth_axis(atm.LONGITUDE, 1)
		
		
		pressure_bar_loc = self._array_with_nth_axis(atm.P, 3)/1E5
		height_km_loc = self._array_with_nth_axis(atm.H, 3)/1E3
		temp_k_loc = self._array_with_nth_axis(atm.T, 3)
		gas_vol_mix_ratio_loc = self._array_with_nth_axis(atm.VMR, 4)
		aerosol_species_density_loc = self._array_with_nth_axis(atm.DUST, 4)

		
		# fig info
		fig_spec = dict(
			figsize=(3*(3*atm.NLOCATIONS),12)
		)
		
		# subfig info
		subfig_spec = dict(
			nrows=atm.NLOCATIONS, 
			ncols=1, 
			squeeze=False
		)
		
		# subplot info
		subplot_spec = dict(
			nrows=3, 
			ncols=3, 
			squeeze=False
		)
		
		
		f = plt.figure(**fig_spec) if f is None else f
		subfigs = f.subfigures(**subfig_spec).flatten()
		axes = tuple(subfig.subplots(**subplot_spec).flatten() for subfig in subfigs)
		
		# At this point, we have figure, subfigure, and axes
		

		#f.suptitle('All atmospheric profiles')

		for k in range(atm.NLOCATIONS):
			subfig = subfigs[k]
			ax_iter = iter(axes[k])
		
			legend_top_left_corner = None
		
			# get values for each location
			latitude = latitude_loc[...,k]
			longitude = longitude_loc[...,k]
			pressure_bar = pressure_bar_loc[...,k]
			height_km = height_km_loc[...,k]
			temp_k = temp_k_loc[...,k]
			gas_vol_mix_ratio = gas_vol_mix_ratio_loc[...,k]
			aerosol_species_density = aerosol_species_density_loc[...,k]

			subfig.suptitle(f'Atmospheric profiles at location index = {k} lat = {latitude} lon = {longitude}')

			# Plot pressure vs height
			with Value(next(ax_iter)) as ax:
				ax.set_xlabel('pressure (bar)')
				ax.set_ylabel('height (km)')
				ax.semilogx(pressure_bar, height_km, c='black', alpha=0.6)
				ax.axvline(1, c='black', ls=':', alpha=0.3)
				ax.axhline(0, c='black', ls=':', alpha=0.3)
				ax.grid(alpha=0.3)


			# Leave some free space for the legend
			with Value(next(ax_iter)) as ax:
				legend_top_left_corner = (ax.get_position().xmin, ax.get_position().ymax)
				ax.remove()
				# do nothing


			with Value(next(ax_iter)) as ax:
				ax.remove()
				# do nothing



			## HEIGHT ##

			with Value(next(ax_iter)) as ax:
				ax.set_ylabel('height (km)')
				ax.set_xlabel(r'aerosol_density (particles $m^{-3}$)')
				for i in range(atm.NDUST):
					label = label = f'aerosol_species_id {i+1} - ' + aerosol_density_models[i].__class__.__name__ if aerosol_density_models[i] is not None else 'static profile'
					ax.plot(aerosol_species_density[:,i], height_km, alpha=0.6, label = label)
					#ax.semilogx(aerosol_species_density[:,i], height_km, alpha=0.6, label = label)

				ax.axhline(0, c='black', ls=':', alpha=0.3)
				ax.grid(alpha=0.3)


			with Value(next(ax_iter)) as ax:
				#ax.set_ylabel('height (km)')
				ax.set_xlabel('volume mixing ratio')
				for i in range(atm.NVMR):
					gas_id_info = ans.Data.gas_info[str(atm.ID[i])]
					ax.semilogx(gas_vol_mix_ratio[:,i], height_km, ls='--', alpha=0.6, label= f'{gas_id_info["name"]} isotope {atm.ISO[i]} - {gas_vmr_models[i].__class__.__name__ if gas_vmr_models[i] is not None else "static profile"}')
					#ax.loglog(gas_vol_mix_ratio[:,i], height_km, ls='--', alpha=0.6, label= f'{gas_id_info['name']} isotope {atm.ISO[i]} - {gas_vmr_models[i].name if gas_vmr_models[i] is not None else 'static profile'}')

				ax.axhline(0, c='black', ls=':', alpha=0.3)
				ax.grid(alpha=0.3)
				#ax.legend()

			with Value(next(ax_iter)) as ax:
				#ax.set_ylabel('height (km)')
				ax.set_xlabel('Temperature (K)')
				ax.semilogx(temp_k, height_km, ls=':', alpha=0.6, label='temperature' )
				ax.axhline(0, c='black', ls=':', alpha=0.3)
				ax.grid(alpha=0.3)

			## PRESSURE ##

			with Value(next(ax_iter)) as ax:
				ax.set_ylabel('pressure (bar)')
				ax.set_xlabel(r'aerosol_density (particles $m^{-3}$)')
				for i in range(atm.NDUST):
					#ax.semilogy(aerosol_species_density[:,i], pressure_bar, alpha=0.6)
					ax.loglog(aerosol_species_density[:,i], pressure_bar, alpha=0.6)

				ax.axhline(1, c='black', ls=':', alpha=0.3)
				ax.invert_yaxis()
				ax.grid(alpha=0.3)


			with Value(next(ax_iter)) as ax:
				#ax.set_ylabel('pressure (bar)')
				ax.set_xlabel('volume mixing ratio')
				for i in range(atm.NVMR):
					gas_id_info = ans.Data.gas_info[str(atm.ID[i])]
					ax.loglog(gas_vol_mix_ratio[:,i], pressure_bar, ls='--', alpha=0.6)

				ax.axhline(1, c='black', ls=':', alpha=0.3)
				ax.invert_yaxis()
				ax.grid(alpha=0.3)
				#ax.legend()


			with Value(next(ax_iter)) as ax:
				#ax.set_ylabel('pressure (bar)')
				ax.set_xlabel('Temperature (K)')
				ax.loglog(temp_k, pressure_bar, ls=':', alpha=0.6)
				ax.axhline(1, c='black', ls=':', alpha=0.3)
				ax.invert_yaxis()
				ax.grid(alpha=0.3)
				#ax.legend()

			if legend_top_left_corner is None:
				subfig.legend(loc = 'upper left', bbox_to_anchor=(0.5, 0.89))
			else:
				subfig.legend(loc = 'upper left', bbox_to_anchor=legend_top_left_corner)
		
		return f, axes