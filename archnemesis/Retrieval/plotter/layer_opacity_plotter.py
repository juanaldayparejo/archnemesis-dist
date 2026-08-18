


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
class LayerOpacityPlotter:
	retrieval : "Retrieval"
	
	def can_plot(self):
		return self.retrieval.Layer is not None
	
	def plot(self) -> tuple[str, mpl.figure.Figure, tuple[np.ndarray[mpl.axes.Axes]]]:
		if self.retrieval.Layer.TAUTOT is None:
			self.retrieval.calculate_layer_opacity()
		return 'layer_opacity', *self.do_plot(self.retrieval.Layer, self.retrieval.Spectroscopy, self.retrieval.Measurement)

	def do_plot(self, Layer, Spectroscopy, Measurement, f=None) -> tuple[mpl.figure.Figure, tuple[np.ndarray[mpl.axes.Axes]]]:
		
		_lgr.debug(f'{Spectroscopy.WAVE.shape=} {Spectroscopy.NWAVE}')
		_lgr.debug(f'{Layer.BASEP.shape=}')
		_lgr.debug(f'{Layer.TAUTOT.shape=}')
	
		#Calculating new wave array
		#Measurement.build_ils(IGEOM=0)
		#wavecalc_min,wavecalc_max = Measurement.calc_wave_range(apply_doppler=True,IGEOM=0)
			
		#Reading tables in the required wavelength range
		#Spectroscopy.read_tables(wavemin=wavecalc_min,wavemax=wavecalc_max)
	
		# TAUTOT : (NWAV,NG,NLAYER), DELG : (NG) -> opacity_at_wav_layer : (NWAV,NLAYER)
		opacity_at_wav_layer = np.tensordot(Layer.TAUTOT, Spectroscopy.DELG, axes=([1],[0])).T[:,::-1] # opacity is unitless, is already scaled to thickness of layer
		
		wav_edges = np.array([
			1.5*Spectroscopy.WAVE[0] - 0.5*Spectroscopy.WAVE[1],
			*(0.5*(Spectroscopy.WAVE[:-1] + Spectroscopy.WAVE[1:])),
			1.5*Spectroscopy.WAVE[-1] - 0.5*Spectroscopy.WAVE[-2]
		])
		
		pressure_edges = (np.array([*Layer.BASEP, 2*Layer.BASEP[-1]-Layer.BASEP[-2]]) / 1E5)
		height_edges = (np.array([*Layer.BASEH, 2*Layer.BASEH[-1]-Layer.BASEH[-2]]) / 1E3)
		
		height_to_pressure = lambda h: np.exp(np.interp(h, height_edges, np.log(pressure_edges)))
		pressure_to_height = lambda p: np.log(np.interp(p, pressure_edges[::-1], np.exp(height_edges[::-1])))
		
		# NOTE: assume light is coming from "top" and heading "downwards"
		#cumulative_opacity = np.cumsum((opacity_at_wav_layer.T*np.diff(height_edges)).T[::-1], axis=0)[::-1]
		downwards_cumulative_opacity = np.cumsum(opacity_at_wav_layer[::-1], axis=0)[::-1]
		#upwards_cumulative_opacity = np.cumsum(opacity_at_wav_layer, axis=0)
		
		downwards_transmission_at_each_layer = np.exp(-opacity_at_wav_layer)
		scattering_or_absorption_at_each_layer = 1 - downwards_transmission_at_each_layer
		cumulative_downwards_transmission_at_each_layer = np.exp(-downwards_cumulative_opacity)
		#cumulative_upwards_transmission_at_each_layer = np.exp(-upwards_cumulative_opacity)
		
		# Assuming:
		# 1) Low Temperature Thermal Equilibrium - i.e., all absorbed energy is re-emitted at a wavelength outside the range we are looking at
		# 2) Opacity, Scattering, and Absorption is isotropic - i.e., bulk media consists of things with a uniform distribution of directions.
		#    Therefore interaction is not dependent on the direction of **incident** light.
		#
		# v - wavelength
		# extinction(v) = scattering(v) + absorption(v)
		# NOTE: I am using `extinction` to be the total attenuation coefficient.
		# transmittance(v) = np.exp[-extinction(v)] = np.exp[-scattering(v)]*np.exp[-absoption(v)]
		#
		# NOTE: Rayleigh scattering varys as 1+cos^2(theta), with theta = scattering angle. This is symmetric around PI, so
		#       backscatter and forward scatter are the same.
		#
		#
		# To do this properly I should do a proper radiative transfer solution. However I can think of some limiting cases:
		#
		# 1) reflectance (backscatter) is a constant fraction of the light that gets to a level. The maximum fraction of light
		#    that can be reflected at a level is the minimum value of `1-transmittance`. This will therefore be a wavelength-independent
		#    continuum reflectance at each level.
		#
		# 2) reflectance (backscatter) is a constant fraction of the extinction coefficient. Therefore the
		#    square of the downwards transmission multiplied by the `1-transmittance` will be the reflectance
		#    of this approximation up to a factor.
		#
		# 3) A combination of case (1) and (2).
		
		transmittance_from_top_to_layer_to_top = cumulative_downwards_transmission_at_each_layer**2
		
		continuum_extinction_at_each_layer = np.min(scattering_or_absorption_at_each_layer, axis=1)[:,None]*np.ones_like(opacity_at_wav_layer)
		
		reflectance_1_constant_fraction_of_light = continuum_extinction_at_each_layer*transmittance_from_top_to_layer_to_top
		reflectance_2_constant_fraction_of_extinction_coefficient = scattering_or_absorption_at_each_layer*transmittance_from_top_to_layer_to_top
		
		reflectance_equal_parts_1_2 = (reflectance_1_constant_fraction_of_light + reflectance_2_constant_fraction_of_extinction_coefficient)/2
		
		
		#print(f'{height_edges=}')
		
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
		
		# subplot info
		subplot_spec = dict(
			nrows=3, 
			ncols=3,
			squeeze=False,
			sharey=True,
			sharex=True
		)
		
		# style info
		#hline_style = dict(
		#	linestyle = '--',
		#	linewidth=0.5,
		#	alpha=0.6,
		#	color='black',
		#)
		
		#plotline_style = dict(
		#	marker='x',
		#	markersize=4,
		#	linewidth=1,
		#	alpha=0.6,
		#)
		
		
		
		
		f = plt.figure(**fig_spec) if f is None else f
		subfigs = f.subfigures(**subfig_spec).flatten()
		axes = tuple(subfig.subplots(**subplot_spec).flatten() for subfig in subfigs)
	
		minmax_str = lambda x: f'[{np.nanmin(x):07.2E}, {np.nanmax(x):07.2E}]'
		
		for subfig, sf_axes in zip(subfigs, axes):
			if Layer.BASEH is None:
				_lgr.warning('Cannot create plot of Layer opacity as they have not been calculated yet')
				subfig.suptitle('Layers have not been calculated yet')
				for ax in sf_axes:
					ax.remove()
				return f, axes
			else:
				subfig.suptitle(f'Layer Opacities\nnumber_of_layers = {Layer.NLAY}')
				sec_yax = sf_axes[0].secondary_yaxis('right', functions=(height_to_pressure, pressure_to_height))
				sec_yax.set_label('Pressure (bar)')
				for ax in sf_axes[1:]:
					ax.sharey(sf_axes[0])
					sec_yax = ax.secondary_yaxis('right', functions=(height_to_pressure, pressure_to_height))
					sec_yax.set_label('Pressure (bar)')
				
			ax_iter = iter(sf_axes)
			
			with Value(next(ax_iter)) as ax:
				ax.set_title(f'log(Opacity) of each layer\n{minmax_str(np.log(opacity_at_wav_layer))}')
				_ = ax.pcolormesh(
					wav_edges,
					height_edges,
					np.log(opacity_at_wav_layer),
				)
				#ax.set_xlabel('spectral axis')
				ax.set_ylabel('Height (km)')
			
			
			with Value(next(ax_iter)) as ax:
				ax.set_title(f'log(Downwards Cumulative Opacity)\n{minmax_str(np.log(downwards_cumulative_opacity))}')
				_ = ax.pcolormesh(
					wav_edges,
					height_edges,
					np.log(downwards_cumulative_opacity),
				)
				#ax.set_xlabel('spectral axis')
				#ax.set_ylabel('Height (km)')
			
			with Value(next(ax_iter)) as ax:
				ax.set_title(f'Downwards Transmission of each layer\n{minmax_str(downwards_transmission_at_each_layer)}')
				_ = ax.pcolormesh(
					wav_edges,
					height_edges,
					downwards_transmission_at_each_layer,
				)
				#ax.set_xlabel('spectral axis')
				#ax.set_ylabel('Height (km)')
			
			with Value(next(ax_iter)) as ax:
				ax.set_title(f'Cumulative Downwards Transmission\n{minmax_str(cumulative_downwards_transmission_at_each_layer)}')
				_ = ax.pcolormesh(
					wav_edges,
					height_edges,
					cumulative_downwards_transmission_at_each_layer,
				)
				#ax.set_xlabel('spectral axis')
				ax.set_ylabel('Height (km)')
			
			with Value(next(ax_iter)) as ax:
				ax.set_title(f'Round trip transmision\n{minmax_str(transmittance_from_top_to_layer_to_top)}')
				_ = ax.pcolormesh(
					wav_edges,
					height_edges,
					transmittance_from_top_to_layer_to_top,
				)
				#ax.set_xlabel('spectral axis')
				#ax.set_ylabel('Height (km)')
			
			with Value(next(ax_iter)) as ax:
				ax.set_title(f'continuum extinction by layer\n{minmax_str(continuum_extinction_at_each_layer)}')
				_ = ax.pcolormesh(
					wav_edges,
					height_edges,
					continuum_extinction_at_each_layer,
				)
				#ax.set_xlabel('spectral axis')
				#ax.set_ylabel('Height (km)')
			
			with Value(next(ax_iter)) as ax:
				ax.set_title(f'reflectance_1\ncontinuum reflection only\n{minmax_str(reflectance_1_constant_fraction_of_light)}')
				_ = ax.pcolormesh(
					wav_edges,
					height_edges,
					reflectance_1_constant_fraction_of_light,
				)
				ax.set_xlabel('spectral axis')
				ax.set_ylabel('Height (km)')
			
			with Value(next(ax_iter)) as ax:
				ax.set_title(f'reflectance_2\nonly reflection, no absorbtion\n{minmax_str(reflectance_2_constant_fraction_of_extinction_coefficient)}')
				_ = ax.pcolormesh(
					wav_edges,
					height_edges,
					reflectance_2_constant_fraction_of_extinction_coefficient,
				)
				ax.set_xlabel('spectral axis')
				#ax.set_ylabel('Height (km)')
			
			with Value(next(ax_iter)) as ax:
				ax.set_title(f'reflectance_equal_parts_1_2\n{minmax_str(reflectance_equal_parts_1_2)}')
				_ = ax.pcolormesh(
					wav_edges,
					height_edges,
					reflectance_equal_parts_1_2,
				)
				ax.set_xlabel('spectral axis')
				#ax.set_ylabel('Height (km)')
			
			
			for ax in ax_iter: # Remove unused axes
				ax.remove()
		
		return f, axes
	