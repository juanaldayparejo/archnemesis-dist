


import dataclasses as dc
from typing import TYPE_CHECKING

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt

from archnemesis.enum import WaveUnitEnum

from .value import Value

if TYPE_CHECKING:
	from ..Retrieval import Retrieval

import archnemesis.cfg.logs as logging
_lgr = logging.getLogger(__name__)
_lgr.setLevel(logging.INFO)



@dc.dataclass(slots=True)
class MeasurementPlotter:
	retrieval : "Retrieval"
	
	
	def can_plot(self):
		return self.retrieval.Measurement is not None
	
	def plot(self) -> tuple[str, mpl.figure.Figure, tuple[np.ndarray[mpl.axes.Axes]]]:
		return 'measurement', *self.do_plot(self.retrieval.Measurement)

	def do_plot(self, Measurement, n_error_sigma = 3, f = None) -> tuple[mpl.figure.Figure, tuple[np.ndarray[mpl.axes.Axes]]]:
		from mpl_toolkits.basemap import Basemap
		from mpl_toolkits.axes_grid1 import make_axes_locatable

		Measurement.SUBOBS_LAT = Measurement.SUBOBS_LAT if Measurement.SUBOBS_LAT is not None else Measurement.LATITUDE
		Measurement.SUBOBS_LON = Measurement.SUBOBS_LON if Measurement.SUBOBS_LON is not None else Measurement.LONGITUDE

		# fig info
		fig_spec = dict(
			figsize=(12, 7)
		)
		
		# subfig info
		subfig_spec = dict(
			nrows=Measurement.NGEOM, 
			ncols=1, 
			squeeze=False
		)
		
		# subplot mosaic info
		subplot_layout = [
			['geometry', 'spectra_linear'],
			['geometry', 'spectra_log'],
		]
		
		
		f = plt.figure(**fig_spec) if f is None else f
		subfigs = f.subfigures(**subfig_spec).flatten()
		axes = tuple(subfig.subplot_mosaic(subplot_layout) for subfig in subfigs)
		
		_lgr.debug(f'{axes = }')

		

		#Making a figure for each geometry
		for igeom in range(Measurement.NGEOM):
			subfig = subfigs[igeom]
			# set axes sharing
			axes[igeom]['spectra_log'].sharex(axes[igeom]['spectra_linear'])
			
			ax_iter = iter(axes[igeom].values())
			
			
			subfig.suptitle('\n'.join((
				f'Geometry index = {igeom}',
				f'Sub-Observer (lat, lon) = ({Measurement.SUBOBS_LAT:+6.2f}, {Measurement.SUBOBS_LON:+6.2f}) [Center of Planet Disk]',
				f'Field of View (LAT, LON) = ({Measurement.LATITUDE:+6.2f}, {Measurement.LONGITUDE:+6.2f}) [Red Circle]',
				f'Number of FOV Averaging Points = {Measurement.NAV[igeom]} [Coloured Dots]'
			)))
			
			spec_slice = slice(0,Measurement.NCONV[igeom])
			
			flons = Measurement.FLON[igeom,:Measurement.NAV[igeom]] if Measurement.NAV[igeom] > 1 else Measurement.FLON[None,:]
			flats = Measurement.FLAT[igeom,:Measurement.NAV[igeom]] if Measurement.NAV[igeom] > 1 else Measurement.FLAT[None,:]
			fweights = Measurement.WGEOM[igeom,:Measurement.NAV[igeom]] if Measurement.NAV[igeom] > 1 else Measurement.WGEOM[None,:]
			
			with Value(next(ax_iter)) as ax:
				#Plotting the geometry
				#ax = plt.subplot2grid((2,3),(0,0),rowspan=2,colspan=1)
				_lgr.debug(f'{ax = }')
				_lgr.debug(f'{Measurement.SUBOBS_LAT=}')
				_lgr.debug(f'{Measurement.SUBOBS_LON=}')
				_lgr.debug(f'{Measurement.LATITUDE=}')
				_lgr.debug(f'{Measurement.LONGITUDE=}')
				_lgr.debug(f'{Measurement.FLON=}')
				_lgr.debug(f'{Measurement.FLAT=}')

				map = Basemap(
					projection='ortho', 
					resolution=None,
					lat_0=Measurement.SUBOBS_LAT, 
					lon_0=Measurement.SUBOBS_LON, 
					ax=ax
				)

				
				map.drawparallels(np.linspace(-90, 90, 13)) # lats
				map.drawmeridians(np.linspace(-180, 180, 13)) # lons


				im = map.plot(
					Measurement.LONGITUDE, 
					Measurement.LATITUDE, 
					latlon=True, 
					marker='o',
					markersize=10,
					markerfacecolor='none',
					markeredgecolor='tab:red',
					markeredgewidth=1,
					linestyle='none'
				)


				im = map.scatter(
					flons[igeom,:Measurement.NAV[igeom]], 
					flats[igeom,:Measurement.NAV[igeom]], 
					latlon=True, 
					c=fweights,
					marker='.'
				)

				if Measurement.NAV[igeom]>1:
					divider = make_axes_locatable(ax)
					cax = divider.append_axes("bottom", size="5%", pad=0.15)
					cbar2 = plt.colorbar(im,cax=cax,orientation='horizontal')
					cbar2.set_label('Weight')
				

				ax.set_title('Geometry '+str(igeom+1))
			
			
			with Value(next(ax_iter)) as ax:
				#Plotting the spectra in linear scale
				#ax = next(ax_iter)
				#ax = plt.subplot2grid((2,3),(0,1),rowspan=1,colspan=2)
				_lgr.debug(f'{ax = }')
				
				ax.fill_between(
					Measurement.VCONV[spec_slice,igeom],
					Measurement.MEAS[spec_slice,igeom] - n_error_sigma*Measurement.ERRMEAS[spec_slice,igeom],
					Measurement.MEAS[spec_slice,igeom] + n_error_sigma*Measurement.ERRMEAS[spec_slice,igeom],
					alpha=0.3,
					label=f'measurement error ({n_error_sigma}'+r'$\sigma$)'
				)
				ax.plot(
					Measurement.VCONV[spec_slice,igeom],
					Measurement.MEAS[spec_slice,igeom],
					alpha=0.6,
					linewidth=1,
					label='measured spectrum'
				)
				
				if Measurement.SPECMOD is not None:
					ax.plot(
						Measurement.VCONV[spec_slice,igeom],
						Measurement.SPECMOD[spec_slice,igeom],
						alpha=0.6,
						linewidth=1,
						label='modelled spectrum'
					)
				
				ax.set_title('Spectra linear scale')
				match WaveUnitEnum(Measurement.ISPACE):
					case WaveUnitEnum.Wavenumber_cm:
						ax.set_ylabel(r'Radiance (W cm$^{-2}$ sr$^{-1}$ (cm$^{-1}$)$^{-1}$)')
					case WaveUnitEnum.Wavelength_um:
						ax.set_ylabel(r'Radiance (W cm$^{-2}$ sr$^{-1}$ $\mu$m$^{-1}$)')
				ax.grid()
				ax.legend()
			

			with Value(next(ax_iter)) as ax:
				#Plotting the spectra in log scale
				#ax = next(ax_iter)
				#ax3 = plt.subplot2grid((2,3),(1,1),rowspan=1,colspan=2,sharex=ax2)
				_lgr.debug(f'{ax = }')
				
				ax.fill_between(
					Measurement.VCONV[spec_slice,igeom],
					Measurement.MEAS[spec_slice,igeom] - n_error_sigma*Measurement.ERRMEAS[spec_slice,igeom],
					Measurement.MEAS[spec_slice,igeom] + n_error_sigma*Measurement.ERRMEAS[spec_slice,igeom],
					alpha=0.3,
				)
				ax.plot(
					Measurement.VCONV[spec_slice,igeom],
					Measurement.MEAS[spec_slice,igeom],
					alpha=0.6,
					linewidth=1,
				)
				
				if Measurement.SPECMOD is not None:
					ax.plot(
						Measurement.VCONV[spec_slice,igeom],
						Measurement.SPECMOD[spec_slice,igeom],
						alpha=0.6,
						linewidth=1,
					)
				ax.set_title('Spectra log scale')
				ax.set_yscale('log')
				match WaveUnitEnum(Measurement.ISPACE):
					case WaveUnitEnum.Wavenumber_cm:
						ax.set_xlabel(r'Wavenumber (cm$^{-1}$)')
						ax.set_ylabel(r'Radiance (W cm$^{-2}$ sr$^{-1}$ (cm$^{-1}$)$^{-1}$)')
					case WaveUnitEnum.Wavelength_um:
						ax.set_xlabel(r'Wavelength ($\mu$m)')
						ax.set_ylabel(r'Radiance (W cm$^{-2}$ sr$^{-1}$ $\mu$m$^{-1}$)')
				ax.grid()
		
		return f, axes
	