import sys, os, os.path
from pathlib import Path
import dataclasses as dc
from typing import Any, Callable, ClassVar, Self
from io import StringIO



import numpy as np
from numpy.lib.stride_tricks import sliding_window_view
import matplotlib.pyplot as plt

from mpl_toolkits.basemap import Basemap

import ltda
import ltda.fits_header
from ltda.io_helper import write_kv, read_kv
from ltda.archnemesis_helper import redirect_file_access
from ltda.archnemesis_helper.retrieval import ArchNemesisFileType, Value, ArchNemesisSpectralUnit
from ltda import archnemesis_helper as anh
from ltda.extensions.pathlib_extensions import PathFormat


import archnemesis.cfg.logs as logging
_lgr = logging.getLogger(__name__)
_lgr.setLevel(logging.DEBUG)


class RetrievalSetParameters:

	@classmethod
	def from_file(
			cls,
			fpath : str | Path
		) -> Self:
		instance = cls()
		instance.read(fpath)
		return instance
		
		
	def __init__(
			self, 
			**kwargs
		):
		self._header = 'This file contains parameters for the retrieval set element.'
		self._entries = kwargs
	
	def set_header(self, header : str) -> Self:
		self._header = header
		return self
	
	def get_header(self) -> str:
		return self._header
	
	def __len__(self)->int:
		return len(self._entries)
	
	def __repr__(self)->str:
		output = StringIO()
		result = None
		for k,(v,c) in self._entries.items():
			write_kv(output, k, v, c)
		result = output.getvalue()
		output.close()
		return result
	
	def __getitem__(self, key) -> tuple[Any, str]:
		"""
		Typically we only want the value part if we are using this interface
		"""
		return (self._entries[key][0])
	
	def __setitem__(self, key, value : Any | tuple[Any, str]) -> None:
		if type(value) == tuple and len(value) == 2 and type(value[1]) == str:
			# We have a (value, description) pair
			self._entries[key] = value
			return
		
		# Otherwise we have a value
		self.set_value(key, value)
		
	
	def add(self, name : str, value : Any, description : None | str = None) -> Self:
		self._entries[name] = (value, description)
		return self
	
	def items(self):
		yield from self._entries.items()
	
	def get_item(self, key) -> tuple[Any, str]:
		return self._entries[key]
	
	def set_item(self, key, value : tuple[Any, str]) -> None:
		if type(value) == tuple and len(value) == 2 and type(value[1]) == str:
			self._entries[key] = value
		else:
			raise RuntimeError("An item of a RetrievalSetParameters instance must be of the form Tuple[Value, Description] (value := Any, description := string).")
	
	def values(self):
		yield from ((k,v[0]) for k,v in self._entries.items())
	
	def descriptions(self):
		yield from ((k,v[1]) for k,v in self._entries.items())
	
	def set_value(self, key, value):
		self._entries[key][0] = value
	
	def set_description(self, key, desc):
		self._entries[key][1] = desc
		
	def write(self, fpath : str | Path):
		with open(fpath, 'w') as f:
			# Write header lines
			f.write(self._header)
			
			# End header with string
			f.write('\nEND HEADER\n')
			
			# Empty lines are ignored, but are useful for formatting purposes
			f.write('\n')
			
			# parameters are written as:
			# key : type = value
			for key, (value, description) in self.items():
				write_kv(f, key, value, comment=description)
		return
	
	def read(self, fpath : str | Path) -> Self:
		header = ''
		
		with open(fpath, 'r') as f:
			line = f.readline()
			n= len(line)
			while n > 0 and (not line.startswith('END HEADER')):
				header += line
				line = f.readline()
				n = len(line)
			
			if n == 0:
				raise RuntimeError(f'Unexpected end of file while reading "{f.name}"')
			
			self.set_header(header)
			
			result = read_kv(f, skip_empty_lines=True)
			
			while result is not None: # None means we are at EOF
				key, vtype, value, comment = result
				self.add(key, value, comment if len(comment) > 0 else None)
			
				result = read_kv(f, skip_empty_lines=True)
		
		return self


@dc.dataclass
class RetrievalSetElement:
	retrieval : anh.Retrieval
	parameters : RetrievalSetParameters
	
	# private attributes
	_param_fname : str = 'retrieval_set_element.params' # Filename used when writing parameters to a set element's directory
	
	@classmethod
	def from_dir(
			cls,
			element_dir : str | Path, 
			path_redirects : tuple[redirect_file_access.Redirector] = tuple(),
			param_fname : str = 'retrieval_set_element.params',
			**kwargs # forwarded to Retrieval method
		) -> Self:
		
		instance = cls(
			anh.Retrieval.from_dir(element_dir, path_redirects=path_redirects, **kwargs),
			RetrievalSetParameters.from_file(Path(element_dir) / param_fname),
			_param_fname = param_fname,
		)
		
		return instance

	
	def write(
			self, 
			filetype : ArchNemesisFileType = ArchNemesisFileType.UNDEFINED,
		) -> None:
		self.retrieval.write(filetype)
		self.parameters.write(Path(self.retrieval.working_directory) / self._param_fname)
	



@dc.dataclass
class RetrievalSet:
	element_param_fname : ClassVar[str] = 'retrieval_set_element.params' # Filename used when writing parameters to a set element's directory
	
	template_path : Path | str # Path to a retrieval that will act as a template (has all the default values) for this set
	containing_dir : Path | str # Path to a directory that will contain all directories of the retrieval set
	
	path_redirects : tuple[redirect_file_access.Redirector] = tuple()
	
	READONLY : bool = False
	
	# private attributes
	_retrieval_set_elements : list[RetrievalSetElement] = dc.field(default_factory=list, init=False, repr=False)
	
	
	@classmethod
	def from_dir(
			cls, 
			containing_dir : Path, 
			READONLY=False,
			path_redirects : tuple[redirect_file_access.Redirector] = tuple(),
			**kwargs # forwarded to Retrieval method
		) -> Self:
		
		_lgr.debug(f'{containing_dir=}')
		_lgr.debug(f'{READONLY=}')
		_lgr.debug(f'{kwargs=}')
		
		# open template file
		with open(Path(containing_dir) / 'template.txt', 'r') as f:
			ident, value = f.readline().split('=')
		
		assert ident.strip() == 'template : Path'
		template_path = Path(value.strip())
		
		instance = cls(template_path, Path(containing_dir).resolve(), READONLY=READONLY, path_redirects=path_redirects)
		
		for element_dir in sorted(filter(lambda x: instance.path_is_a_retrieval_set_element(instance.containing_dir / x), os.listdir(instance.containing_dir))):
			instance._retrieval_set_elements.append(
				RetrievalSetElement.from_dir(instance.containing_dir / element_dir, path_redirects=path_redirects, **kwargs)
			)
			
			_lgr.debug(f'{instance._retrieval_set_elements[-1].retrieval.working_directory=}')
			_lgr.debug(f'{instance._retrieval_set_elements[-1].parameters=}')
		
		return instance
	
	def path_is_a_retrieval_set_element(self, path : Path) -> bool:
		"""
		Determines if a `path` is an element of the retrieval set
		"""
		return path.is_dir() and (path.parent == self.containing_dir) and (path / self.element_param_fname).is_file()
		
	
	def __post_init__(self):		
		if type(self.template_path) is str:
			self.template_path = Path(self.template_path)
		
		if type(self.containing_dir) is str:
			self.containing_dir = Path(self.containing_dir)
	
		self._template = None
		self._retrieval_set_elements : list[RetrievalSetElement] = []
	
		
		if self.template_path.exists():
			# Load the template retrieval if we can
			if not self.template_path.is_file():
				files = os.listdir(self.template_path)
				legacy_input_files = tuple(filter(lambda x: x.endswith('.inp'), files))
				hdf5_input_files = tuple(filter(lambda x: x.endswith('.h5'), files))
				
				if len(hdf5_input_files) > 0:
					self.template_path = Path(self.template_path) / hdf5_input_files[0]
				elif len(legacy_input_files) >0:
					self.template_path = Path(self.template_path) / legacy_input_files[0]
				else:
					raise ValueError('template_path must be a file or a directory containing a *.h5 or *.inp file')
				
			self._template : anh.Retrieval = anh.Retrieval.from_file(self.template_path, path_redirects=self.path_redirects)
		else:
			self.READONLY = True
		
		_lgr.debug(f'{self.template_path=}')
		_lgr.debug(f'{self.containing_dir=}')
		_lgr.debug(f'{self.element_param_fname=}')
		_lgr.debug(f'{self._template=}')
		_lgr.debug(f'{self._retrieval_set_elements=}')
		
		return
	
	@property
	def template(self) -> anh.Retrieval:
		return self._template
	
	@property
	def runname(self) -> str:
		return self.template_path.stem
	
	@property
	def input_file_name(self) -> str:
		return self.template_path.name
	
	@property
	def elements(self) -> tuple[RetrievalSetElement,...]:
		return tuple(self._retrieval_set_elements)
	
	@property
	def size(self) -> int:
		return len(self._retrieval_set_elements)
	
	
	def add_set_element_with_data(
			self, 
			element_data_dict : dict[str,Any], 
			element_params : dict[str,tuple[Any,None|str]] | RetrievalSetParameters,
			element_dir_fmt : str = 'element_{e_idx}',
		) -> None:
		"""
		Adds an element ot the retrieval set by call/writing methods/attributes of the Retrieval specified in `element_data_dict`
		
		## ARGUMENTS ##
			element_data_dict : dict[str,Any]
				A dictionary containing values specific to the element. Will be used to overwrite
				values upon the template retrieval for this specific element. Entries must correspond to
				the following:
					
					1) A key that identifies an attribute (or property) of the retrieval/component of the retrieval,
					where the value is the value that will be assigned to the attribute (or property).
					E.g. {"Measurement.LATITUDE" : 0.0}
					
					2) A key that identifies a method of the retrieval/component of the retrieval, where
					the value is a tuple of arguments that will be passed to the method.
					E.g. {"Atmosphere.edit_DUST" : (np.array([0,0,0,...]),)}
			
			element_params : dict[str,Any]:
				Parameters to write to the `self.element_param_fname` file within the retrieval set element's directory.
				Any `None` entries will be taken from the `element_data_dict` argument.
			
			element_dir_fmt : str = 'element_{e_idx}'
				A "format string" (i.e. will have `.format(...)` called upon it) that is the format for the
				directory we will save the retrieval element into (as a sub-directory of `self.containing_dir`).
				Will recognise the following named values:
					
					e_idx : int 
						The index of the retrieval element, is assigned from zero in order the element is added
						to the set.
					
					e_data : dict[str,Any]
						The `element_data_dict` argument passed to this function
					
					e_params : dict[str,tuple[Any,str|None]]
						The `element_params` argument passed to this function
				
		
		## RETURNS ##
			None
		"""
		if self.READONLY:
			raise RuntimeError('Cannot add a set element as this RetrievalSet is READONLY')
		
		element_params = element_params if (type(element_params) is RetrievalSetParameters) else RetrievalSetParameters(**element_params)
		
		e_idx = len(self._retrieval_set_elements)
		
		for name, value in element_params.values():
			if value is None:
				element_params.set_value(name, element_data_dict[name])
		
		self._retrieval_set_elements.append(
			RetrievalSetElement(
				anh.Retrieval.copy(self._template), 
				element_params
			)
		)
		
		# Ensure that the working directory is set to be within the containing directory
		self._retrieval_set_elements[-1].retrieval.working_directory = self.containing_dir / element_dir_fmt.format(e_idx=e_idx, e_data=element_data_dict, e_params=element_params)
		
		
		
		for key, value in element_data_dict.items():
			attr_target_strs = key.strip().split('.')
			attr_target = self._retrieval_set_elements[-1].retrieval
			
			_last = True
			_n_parts = len(attr_target_strs)
			for i, attr_target_str in enumerate(attr_target_strs):
				_last = i == (_n_parts-1)
				
				if not _last and hasattr(attr_target, attr_target_str):
					attr_target = getattr(attr_target, attr_target_str)
					
				elif _last:
					if hasattr(attr_target, attr_target_str):
						_final_attr_target = getattr(attr_target, attr_target_str)
						if callable(attr_target):
							_final_attr_target(*value)
						else:
							setattr(attr_target, attr_target_str, value)
					else:
						raise AttributeError(f"When trying to set/call `Retrieval.{key}` with `{value}`. Could not find attribute/method `Retrieval.{'.'.join(attr_target_strs[:i])}`.")
				else:
					raise AttributeError(f"When trying to set/call `Retrieval.{key}` with `{value}`. Could not find attribute/method `Retrieval.{'.'.join(attr_target_strs[:i])}`.")
		return
	
	
	def add_set_element_with_callback(
			self,
			callback : Callable[[anh.Retrieval], RetrievalSetElement],
		)->None:
		"""
		Adds an element to the retrieval set by constructing the retrieval via `callback`
		
		## ARGUMENTS ##
		
			callback : Callable[[Retrieval], RetrievalSetElement]
				Callable that accepts the TEMPLATE retrieval and returns a RetrievalSetElement that will be added to the set. 
		
		## RETURNS ##
			None
		"""
		if self.READONLY:
			raise RuntimeError('Cannot add a set element as this RetrievalSet is READONLY')
		
		element = callback(self._template)
		
		# Ensure that the element is relative to the containing directory
		if not element.retrieval.working_directory.resolve().is_relative_to(self.containing_dir.resolve()):
			raise RuntimeError(f'Cannot set an element of a RetrievalSet that has a working directory "{element.retrieval.working_directory.resolve()}" that is not within the containing directory of the RetrievalSet "{self.containing_dir.resolve()}".')
		
		self._retrieval_set_elements.append(element)
		return
	
	
	def create_containing_directory(self, exist_ok=True):
		self.containing_dir.mkdir(exist_ok=exist_ok)
	
	
	def write(
			self, 
			filetype : ArchNemesisFileType = ArchNemesisFileType.UNDEFINED
		) -> None:
		"""
		Write the retrieval set to `containing_dir`, each element of the set will be a sub-directory of `containing_dir`
		
		## ARGUMENTS ##
			filetype : ArchNemesisFileType = ArchNemesisFileType.UNDEFINED
				Type of file to write: HDF5, LEGACY, or UNDEFINED (to write out whatever format was read in)
		
		## RETURNS ##
			None
		"""
		if self.READONLY:
			raise RuntimeError('Cannot write RetrievalSet as this RetrievalSet is READONLY')
		
		# Ensure containing directory exists
		self.create_containing_directory()
		
		# Write the template we are using to the contiaining directory
		with open(self.containing_dir / 'template.txt', 'w') as f:
			f.write(f'template : Path = {self.template_path.resolve()}\n')

		
		
		for element in self._retrieval_set_elements:
			# Ensure that the element is relative to the containing directory
			if not Path(element.retrieval.working_directory).resolve().is_relative_to(Path(self.containing_dir).resolve()):
				raise RuntimeError(f'An element of a RetrievalSet has a working directory "{Path(element.retrieval.working_directory).resolve()}" that is not within the containing directory of the RetrievalSet "{Path(self.containing_dir).resolve()}".')
		
			# Write the retrieval element
			element.write(filetype)
			
		return
	
	
	
	
	def plot_measurement_comparison(
			self, 
			plot_path_fmt : PathFormat |  None = PathFormat('{retrieval_set_dir}/plots/{plot_name}.png'),
			IGNORE_READONLY = False,
			window_size = 1,
		) -> None:
		"""
		Create comparison plots between elements of the retrieval set. Plots will be saved in the 
		`plot_dir` folder (relative to the `RetrievalSet.containing_dir` if the path is relative).
		"""
		
		_lgr.critical('WORK IN PROGRESS')
		
		plot_fpath = None if plot_path_fmt is None else plot_path_fmt.format(self.containing_dir, retrieval_set_dir = self.containing_dir, plot_name='measurement')
		
		if self.READONLY and not IGNORE_READONLY:
			raise RuntimeError('Cannot save plots of this RetrievalSet as this RetrievalSet is READONLY')
		
		
		#n_error_sigma = 1
		window_size = 30
		rolling_average = lambda x, window_size=window_size: np.average(sliding_window_view(x, window_shape = window_size),	axis=-1)
		
		# Assume all retrievals in set share the same number of geometries and spectral units etc.
		
		igeom = 0 # Always use 0th geometry for now
		
		n_elements = len(self.elements)
		n_smoothed_values = self.elements[0].retrieval.Measurement.NCONV[igeom] - (window_size-1)
		n_av = self.elements[0].retrieval.Measurement.NAV[igeom]
		subobs_lat = self.elements[0].retrieval.Measurement.SUBOBS_LAT
		subobs_lon = self.elements[0].retrieval.Measurement.SUBOBS_LON
		
		meas_spec_wavs = np.zeros((n_elements, n_smoothed_values), dtype=float)
		meas_spec_data = np.zeros((n_elements, n_smoothed_values), dtype=float)
		meas_spec_error = np.zeros((n_elements, n_smoothed_values), dtype=float)
		meas_spec_fit = np.zeros((n_elements, n_smoothed_values), dtype=float)
		meas_spec_resid = np.zeros((n_elements, n_smoothed_values), dtype=float)
		
		all_lats = np.zeros((n_elements, 1), dtype=float)
		all_lons = np.zeros((n_elements, 1), dtype=float)
		all_flats = np.zeros((n_elements, 1, n_av), dtype=float)
		all_flons = np.zeros((n_elements, 1, n_av), dtype=float)
		all_fweights = np.zeros((n_elements, 1, n_av), dtype=float)
		
		
		
		spectral_point_label = None
		spectral_radiance_unit = None
		match ArchNemesisSpectralUnit(self.elements[0].retrieval.Measurement.ISPACE):
			case ArchNemesisSpectralUnit.WAVENUMBER_cm:
				spectral_point_label = r'Wavenumber (cm$^{-1}$)'
				spectral_radiance_unit = r'W cm$^{-2}$ sr$^{-1}$ (cm$^{-1}$)$^{-1}$'
			case ArchNemesisSpectralUnit.WAVELENGTH_um:
				spectral_point_label = r'Wavelength ($\mu$m)'
				spectral_radiance_unit = r'W cm$^{-2}$ sr$^{-1}$ $\mu$m$^{-1}$'
		
		spectral_radiance_label = f'Radiance ({spectral_radiance_unit})'
		
		
		for i, element in enumerate(self.elements):
			retrieval = element.retrieval
			#params = element.parameters
					
			## Get values we will plot ##
			
			#retrieval_path_relative_to_set = Path(retrieval.working_directory).resolve().relative_to(self.containing_dir)
			
			Measurement = retrieval.Measurement
		
			spec_slice = slice(0,Measurement.NCONV[igeom])
			
			all_lats[i, igeom:igeom+1] = Measurement.LATITUDE
			all_lons[i, igeom:igeom+1] = Measurement.LONGITUDE
			all_flons[i,igeom:igeom+1] = Measurement.FLON[igeom,:Measurement.NAV[igeom]] if Measurement.NAV[igeom] > 1 else Measurement.FLON[None,:]
			all_flats[i,igeom:igeom+1] = Measurement.FLAT[igeom,:Measurement.NAV[igeom]] if Measurement.NAV[igeom] > 1 else Measurement.FLAT[None,:]
			all_fweights[i,igeom:igeom+1] = Measurement.WGEOM[igeom,:Measurement.NAV[igeom]] if Measurement.NAV[igeom] > 1 else Measurement.WGEOM[None,:]
			
			meas_spec_wavs[i] = rolling_average(Measurement.VCONV[spec_slice, igeom])
			meas_spec_data[i] = rolling_average(Measurement.MEAS[spec_slice,igeom])
			meas_spec_error[i] = rolling_average(Measurement.ERRMEAS[spec_slice,igeom])
			meas_spec_fit[i] = rolling_average(np.full_like(Measurement.MEAS[spec_slice,igeom], fill_value=np.nan) if Measurement.SPECMOD is None else Measurement.SPECMOD[spec_slice,igeom],)
			meas_spec_resid[i] = meas_spec_fit[i] - meas_spec_data[i]
		
		
		
		#apr_style_lw = 1
		#apr_style_lt = ':'
		#apr_style_alpha = 0.6
		
		#fit_style_lw = 1
		#fit_style_lt = '-'
		#fit_style_alpha = 0.6
		
		#resid_style_lw = 1
		#resid_style_lt = '-'
		#resid_style_alpha = 0.6
		
		#err_style_alpha = 0.1
		
		
		
		## Define Figure Parameters ##
		
		# fig info
		fig_spec = dict(
			figsize=(12, 7)
		)
		
		# subfig info
		subfig_spec = dict(
			nrows=1, 
			ncols=1, 
			squeeze=False
		)
		
		# subplot mosaic info
		subplot_layout = [
			['geometry','spectra_linear',],
			['geometry','spectra_residual'],
		]
		
		
		f = plt.figure(**fig_spec)
		subfigs = f.subfigures(**subfig_spec).flatten()
		axes = tuple(subfig.subplot_mosaic(subplot_layout) for subfig in subfigs)
		
		
		f.suptitle('Measurement Comparison\nmeasurement - dotted lines | fit - solid lines | measurement error - coloured regions')
		
		
		
		with Value(axes[0]['spectra_linear']) as ax:
			ax.set_title('Spectra linear scale')
			#ax.set_title('Spectra log scale'); ax.set_yscale('log')
			ax.set_xlabel(spectral_point_label)
			ax.set_ylabel('retrieval index')
				
			im = ax.imshow(
				meas_spec_data,
				extent = (
					np.min(meas_spec_wavs[i]), 
					np.max(meas_spec_wavs[i]),
					0,
					n_elements
				),
				aspect='auto',
			)
			
			ax.get_figure().colorbar(im, ax).set_label(spectral_radiance_label)
		
		with Value(axes[0]['spectra_residual']) as ax:
			ax.set_title('Spectra residual')
			#ax.set_title('Spectra log scale'); ax.set_yscale('log')
			ax.set_xlabel(spectral_point_label)
			ax.set_ylabel('retrieval index')
				
			im = ax.imshow(
				meas_spec_resid,
				extent = (
					np.min(meas_spec_wavs[i]), 
					np.max(meas_spec_wavs[i]),
					0,
					n_elements
				),
				aspect='auto',
			)
			
			ax.get_figure().colorbar(im, ax).set_label(spectral_radiance_label)
			
			
			
		with Value(axes[0]['geometry']) as ax:
			ax.set_title('Geometries\ncirlces - FOV center | dots - contributing points')
			
			map = Basemap(
				projection='ortho', 
				resolution=None,
				lat_0=subobs_lat, 
				lon_0=subobs_lon, 
				ax=ax
			)

			
			map.drawparallels(np.linspace(-90, 90, 13)) #  lats
			map.drawmeridians(np.linspace(-180, 180, 13)) #  lons


			im = map.plot(
				all_lons[:, igeom], 
				all_lats[:, igeom], 
				latlon=True, 
				marker='o',
				markersize=10,
				markerfacecolor='none',
				markeredgecolor='tab:blue',
				markeredgewidth=1,
				linestyle='none'
			)


			im = map.scatter(
				all_flons[:, igeom, :],
				all_flats[:, igeom, :], 
				latlon=True, 
				c='tab:blue',
				marker='.'
			)
				

		ax_iter = iter(axes[0].values())
		for ax in ax_iter:
			ax.grid()
			#ax.legend()
		
		subfigs[0].legend()
		
		if plot_fpath is not None:
			plot_fpath.parent.mkdir(parents=True, exist_ok=True)
			f.savefig(plot_fpath, bbox_inches='tight')
	
	
	def plot_measurement_vs_parameters(
			self, 
			parameters : None | tuple[str] = None,
			plot_path_fmt : PathFormat |  None = PathFormat('{retrieval_set_dir}/plots/{plot_name}.png'),
			IGNORE_READONLY = False,
		) -> None:
		
		plot_fpath = None if plot_path_fmt is None else plot_path_fmt.format(self.containing_dir, retrieval_set_dir = self.containing_dir, plot_name='measurement_vs_params')
		
		parameter_filter = None
		if parameters is None:
			_lgr.info('Plotting against all parameters...')
			parameter_filter = lambda x: tuple(x.items())
		else:
			parameter_filter = lambda x: tuple((k,v) for k,v in x.items() if k in parameters)
		
		
		# Define variables
		pp_vals = dict()
		pp_wavs = dict()
		pp_spec = dict()
		pp_err = dict()
		pp_fit = dict()
		pp_resid = dict()
		
		
		# populate variables
		for retrieval, params in ((e.retrieval, parameter_filter(e.parameters)) for e in self.elements):
			print(f'retrieval=Retrieval(working_directory={Path(retrieval.working_directory).relative_to(self.containing_dir)}) params={tuple(p[0] for p in params)}')
		
			meas = retrieval.Measurement
			
			spec_slice = tuple(slice(0,meas.NCONV[igeom]) for igeom in range(meas.NGEOM))
			

			for param_name, (param_value, param_desc) in params:
				
				vals = pp_vals.get(param_name, list())
				vals.append(param_value)
				pp_vals[param_name] = vals
				
				wavs = pp_wavs.get(param_name, list())
				wavs.append(meas.VCONV[spec_slice])
				pp_wavs[param_name] = wavs
				
				spec = pp_spec.get(param_name, list())
				spec.append(meas.MEAS[spec_slice])
				pp_spec[param_name] = spec
				
				err = pp_err.get(param_name, list())
				err.append(meas.ERRMEAS[spec_slice])
				pp_err[param_name] = err
					
				if meas.SPECMOD is not None:
					fit = pp_fit.get(param_name, list())
					fit.append(meas.SPECMOD[spec_slice])
					pp_fit[param_name] = fit
				else:
					fit = pp_fit.get(param_name, list())
					fit.append(np.full_like(meas.MEAS[spec_slice], fill_value=np.nan))
					pp_fit[param_name] = fit
		
		
		for param_name in pp_vals.keys():
			pp_vals[param_name] = np.array(pp_vals[param_name])
			pp_wavs[param_name] = np.array(pp_wavs[param_name])
			pp_spec[param_name] = np.array(pp_spec[param_name])
			pp_err[param_name] = np.array(pp_err[param_name])
			pp_fit[param_name] = np.array(pp_fit[param_name])
			pp_resid[param_name] = pp_fit[param_name] - pp_spec[param_name]
		
		
		pp_names = tuple(pp_vals.keys())
		_lgr.debug(f'{pp_vals=}')
		
		igeom=0
		
		# do plotting
		## Define Figure Parameters ##
		
		# fig info
		fig_spec = dict(
			figsize=(12, 7)
		)
		
		# subfig info
		subfig_spec = dict(
			nrows=1, 
			ncols=1, 
			squeeze=False
		)
		
		# subplot info
		subplot_spec = dict(
			nrows = len(pp_names), 
			ncols = 3, 
			squeeze=False
		)
		
		
		f = plt.figure(**fig_spec)
		subfigs = f.subfigures(**subfig_spec).flatten()
		sf_axes = tuple(subfig.subplots(**subplot_spec) for subfig in subfigs)
	
		axes = sf_axes[0]
		_lgr.debug(f'{axes.shape=}')
		
	
		for i, axs in enumerate(axes):
			_lgr.debug(f'{axs.shape=}')
			
			param_name = pp_names[i]
			_lgr.debug(f'{param_name=}')
			
			sort_idxs = np.argsort(pp_vals[param_name])

			tx = np.concatenate(
				(
					2*pp_wavs[param_name][sort_idxs,0:1,igeom] - pp_wavs[param_name][sort_idxs,1:2,igeom],
					pp_wavs[param_name][sort_idxs,...,igeom],
					2*pp_wavs[param_name][sort_idxs,-1:-0,igeom] - pp_wavs[param_name][sort_idxs,-2:-1,igeom],
				),
				axis=1
			)
			dx = np.diff(tx)
			x = tx[...,:-1] + dx/2
			
			ty = np.concatenate(
				(
					2*pp_vals[param_name][sort_idxs][0:1] - pp_vals[param_name][sort_idxs][1:2],
					pp_vals[param_name][sort_idxs],
					2*pp_vals[param_name][sort_idxs][-1:-0] - pp_vals[param_name][sort_idxs][-2:-1]
				),
				axis=0
			)
			dy = np.diff(ty)
			y = ty[:-1] + dy/2
			
			with Value(axs[0]) as ax:
				z = pp_spec[param_name][sort_idxs,...,igeom]
				ax.set_title(f'Measured Spectra\n[{np.nanmin(z):07.2E}, {np.nanmax(z):07.2E}]')
				ax.pcolormesh(
					x,
					y,
					z,
				)
				ax.set_xlabel('wave')
				ax.set_ylabel(param_name)
			
			with Value(axs[1]) as ax:
				z = pp_fit[param_name][sort_idxs,...,igeom]
				ax.set_title(f'Fitted Spectra\n[{np.nanmin(z):07.2E}, {np.nanmax(z):07.2E}]')
				ax.pcolormesh(
					x,
					y,
					z,
				)
				ax.set_xlabel('wave')
				ax.set_ylabel(param_name)
			
			with Value(axs[2]) as ax:
				z = pp_resid[param_name][sort_idxs,...,igeom]
				ax.set_title(f'Residual Spectra\n[{np.nanmin(z):07.2E}, {np.nanmax(z):07.2E}]')
				ax.pcolormesh(
					x,
					y,
					z,
					cmap='bwr',
				)
				ax.set_xlabel('wave')
				ax.set_ylabel(param_name)
		
		if plot_fpath is not None:
			plot_fpath.parent.mkdir(parents=True, exist_ok=True)
			f.savefig(plot_fpath, bbox_inches='tight')
		
		return
	
	
	
	def plot_optimal_estimation_comparison(self):
		raise NotImplementedError
	
	def plot_model_parameters_comparison(self):
		raise NotImplementedError
	
	def plot_model_parameters_vs_parameters(
			self, 
			parameters : None | tuple[str] = None,
			plot_path_fmt : PathFormat |  None = PathFormat('{retrieval_set_dir}/plots/{plot_name}.png'),
			IGNORE_READONLY = False,
		) -> None:
		"""
		Plots apriori and retrieved model parameters (from state vector) vs set element parameters
		"""
		
		plot_fpath = None if plot_path_fmt is None else plot_path_fmt.format(self.containing_dir, retrieval_set_dir = self.containing_dir, plot_name='model_parameters_vs_params')
		
		parameter_filter = None
		if parameters is None:
			_lgr.info('Plotting against all parameters...')
			parameter_filter = lambda x: tuple(x.items())
		else:
			parameter_filter = lambda x: tuple((k,v) for k,v in x.items() if k in parameters)
		
		# Define variables
		pp_vals = dict()
		pp_sort_idxs = dict()
		pp_model_param_ids = dict()
		pp_model_param_values = dict()
		pp_model_param_entries = dict()
		
		
		# NOTE: At the moment there is not a way to uniquely identify a model based upon what the model is supposed to represent
		# therefore we will assume that all elements of a set share the same models in the same order.
		
		# populate variables
		for retrieval, params in ((e.retrieval, parameter_filter(e.parameters)) for e in self.elements):
			print(f'retrieval=Retrieval(working_directory={Path(retrieval.working_directory).relative_to(self.containing_dir)}) params={tuple(p[0] for p in params)}')
		
			current_model_params = retrieval.Variables.model_parameters
			
			for param_idx, (param_name, (param_value, param_desc)) in enumerate(params):
				
				vals = pp_vals.get(param_name, list())
				vals.append(param_value)
				pp_vals[param_name] = vals
				
				
				model_param_entries = pp_model_param_entries.get(param_name, list())
				model_param_ids = pp_model_param_ids.get(param_name, list())
				model_param_values = pp_model_param_values.get(param_name, list())
				#model_param_entries.append(current_model_params)
				model_param_entries.append(dict())
				model_param_ids.append([])
				model_param_values.append([])
				for i, cmp in enumerate(current_model_params):
					for j, m_entry in enumerate(cmp.values()):
						for k, (av, pv, fv) in enumerate(zip(m_entry.apriori_value, m_entry.posterior_value, m_entry.is_fixed)):
							model_param_ids[-1].append(((i, m_entry.model_id), (k, m_entry.name)))
							model_param_values[-1].append((av, pv, fv))
							model_param_entries[-1][((i, m_entry.model_id), (k, m_entry.name))] = (av, pv, fv)
				pp_model_param_entries[param_name] = model_param_entries
				pp_model_param_ids[param_name] = model_param_ids
				pp_model_param_values[param_name] = model_param_values
				
		
		for param_name in pp_vals.keys():
			pp_vals[param_name] = np.array(pp_vals[param_name])
			pp_sort_idxs[param_name] = np.argsort(pp_vals[param_name])
			pp_model_param_values[param_name] = np.array(pp_model_param_values[param_name])
			
		_lgr.debug(f'{[x for x in tuple(pp_model_param_values.values())[0][0] if x[2]==False]=}')
			
			#_lgr.debug(f'{param_name=} pp_model_param_entries.shape=({len(pp_model_param_entries[param_name])}, {len(pp_model_param_entries[param_name][0])}, {len(pp_model_param_entries[param_name][0][0])})')
		
		# do plotting
		## Define Figure Parameters ##
		ap_style=dict(
			color='tab:blue',
			alpha=0.3,
			linestyle='-',
			linewidth=1,
			marker='.'
		)
		
		fit_style=dict(
			color='tab:orange',
			alpha=0.6,
			linestyle='-',
			linewidth=1,
			marker='.'
		)
		
		
		#resid_colour = 'tab:red'
		#resid_alpha = 0.3
		#resid_linestyle = '--'
		
		
		n_params = len(pp_vals)
		n_model_params = len([1 for x in tuple(pp_model_param_values.values())[0][0] if x[2]==False]) # only show non-fixed values
		_lgr.debug(f'{n_params=} {n_model_params=}')
		
		# fig info
		fig_spec = dict(
			figsize=(4*n_params, 4*n_model_params)
		)
		
		# subfig info
		subfig_spec = dict(
			nrows=1, 
			ncols=1, 
			squeeze=False
		)
		
		# subplot info
		subplot_spec = dict(
			nrows = n_model_params, 
			ncols = n_params,
			squeeze=False
		)
		
		
		f = plt.figure(**fig_spec)
		subfigs = f.subfigures(**subfig_spec).flatten()
		sf_axes = tuple(subfig.subplots(**subplot_spec) for subfig in subfigs)
	
		axes = sf_axes[0]
		_lgr.debug(f'{axes.shape=}')
		
		subfigs[0].suptitle('model parameters vs retrieval set parameters')
		
		first_flag = True
		
		for i, axs in enumerate(axes.T):
			first_col_flag = i == 0
			last_col_flag = i == (n_params-1)
		
			_lgr.debug(f'{axs.shape=}')
			
			param_name = tuple(pp_vals.keys())[i]
			_lgr.debug(f'{param_name=}')
			
			_lgr.debug(f'{pp_sort_idxs[param_name]=}')
			_lgr.debug(f'{pp_model_param_values[param_name].shape=}')
			
			model_param_values = pp_model_param_values[param_name][pp_sort_idxs[param_name]]
			_lgr.debug(f'{model_param_values.shape=}')
			
			not_fixed_mask = model_param_values[pp_sort_idxs[param_name],:,2]==0
			_lgr.debug(f'{not_fixed_mask.shape=}')
			
			_lgr.debug(f'{len(model_param_ids)=}, {len(model_param_ids[0])}')
			model_param_ids = [pp_model_param_ids[param_name][s] for s in pp_sort_idxs[param_name]] # sort
			model_param_ids = [[b for j,b in enumerate(a) if not_fixed_mask[i,j]] for i, a in enumerate(model_param_ids) ] # select non fixed entries
			model_param_ids = tuple(zip(*model_param_ids))
			_lgr.debug(f'{len(model_param_ids)=}, {len(model_param_ids[0])}')
			
			
			
			
			x = pp_vals[param_name][pp_sort_idxs[param_name]]
			
			_lgr.debug(f'{x.shape=}')
			
			y_apriori = model_param_values[not_fixed_mask,0].reshape(*x.shape,-1).T
			y_posterior = model_param_values[not_fixed_mask,1].reshape(*x.shape,-1).T
			y_residual = y_posterior - y_apriori
			
			_lgr.debug(f'{y_apriori.shape=}')
			_lgr.debug(f'{y_posterior.shape=}')
			_lgr.debug(f'{y_residual.shape=}')
			
			
			
			
			for j, model_param_id_group in enumerate(model_param_ids):
				first_row_flag = j ==0
				last_row_flag = j == (n_model_params-1)
				
				assert all((a == model_param_id_group[0] for a in model_param_id_group)), "Assume that all elements of retrieval set have the same state vector layout."
				
				model_param_id = model_param_id_group[0]
				_lgr.debug(f'{x=}')
				_lgr.debug(f'{y_apriori[j]=}')
				_lgr.debug(f'{y_posterior[j]=}')
				_lgr.debug(f'{y_residual[j]=}')
				_lgr.debug(f'{model_param_id=}')
			
				with Value(axs[j]) as ax:
					#ax.set_title(f'{param_name}\nmodel_index {model_param_id[0][0]}, model_id {model_param_id[0][1]}')
					ax.plot(
						x, 
						y_apriori[j], 
						label='apriori' if first_flag else None,
						**ap_style
					)
					ax.plot(
						x, 
						y_posterior[j], 
						label='posterior' if first_flag else None,
						**fit_style
					)
					#ax2 = ax.twinx()
					#ax2.plot(
					#	x, 
					#	y_residual[j], 
					#	linestyle=resid_linestyle, 
					#	color=resid_colour, 
					#	alpha=resid_alpha, 
					#	label='residual' if first_flag else None
					#)
					
					[a.set_visible(last_row_flag) for a in ax.get_xticklabels(which='both')]
					[a.set_visible(first_col_flag) for a in ax.get_yticklabels(which='both')]
					#[a.set_visible(last_col_flag) for a in ax2.get_yticklabels(which='both')]
					
					if first_row_flag:
						ax.set_title(f'{param_name}')
					
					if last_row_flag:
						ax.set_xlabel(param_name)
					
					if first_col_flag:
						ax.set_ylabel(f'model index : {model_param_id[0][0]}\nmodel id : {model_param_id[0][1]}\n{model_param_id[1][1]}[{model_param_id[1][0]}]')
					
					if last_col_flag:
						pass
						#ax2.set_ylabel('residual')
					
				first_flag = False
		
		subfigs[0].legend()
		
		if plot_fpath is not None:
			plot_fpath.parent.mkdir(parents=True, exist_ok=True)
			f.savefig(plot_fpath, bbox_inches='tight')
		
		return
	
	def plot_fit_stats_vs_parameters(
			self, 
			parameters : None | tuple[str] = None,
			plot_path_fmt : PathFormat |  None = PathFormat('{retrieval_set_dir}/plots/{plot_name}.png'),
			IGNORE_READONLY = False,
		) -> None:
		"""
		Plots apriori and retrieved model parameters (from state vector) vs set element parameters
		"""
		
		plot_fpath = None if plot_path_fmt is None else plot_path_fmt.format(self.containing_dir, retrieval_set_dir = self.containing_dir, plot_name='fit_stats_vs_params')
		
		parameter_filter = None
		if parameters is None:
			_lgr.info('Plotting against all parameters...')
			parameter_filter = lambda x: tuple(x.items())
		else:
			parameter_filter = lambda x: tuple((k,v) for k,v in x.items() if k in parameters)
			
		# Define variables
		pp_vals = dict()
		pp_sort_idxs = dict()
		pp_chisq = dict()
		pp_phi = dict()
		
		
		# NOTE: At the moment there is not a way to uniquely identify a model based upon what the model is supposed to represent
		# therefore we will assume that all elements of a set share the same models in the same order.
		
		# populate variables
		for retrieval, params in ((e.retrieval, parameter_filter(e.parameters)) for e in self.elements):
			print(f'retrieval=Retrieval(working_directory={Path(retrieval.working_directory).relative_to(self.containing_dir)}) params={tuple(p[0] for p in params)}')
		
			retrieval_result = retrieval.OptimalEstimationSetup
			
			for param_idx, (param_name, (param_value, param_desc)) in enumerate(params):
				
				vals = pp_vals.get(param_name, list())
				vals.append(param_value)
				pp_vals[param_name] = vals
				
				chisq = pp_chisq.get(param_name, list())
				chisq.append(retrieval_result.CHISQ)
				pp_chisq[param_name] = chisq
				
				phi = pp_phi.get(param_name, list())
				phi.append(retrieval_result.PHI)
				pp_phi[param_name] = phi
				
		
		for param_name in pp_vals.keys():
			pp_sort_idxs[param_name] = np.argsort(pp_vals[param_name])
			pp_vals[param_name] = np.array(pp_vals[param_name])[pp_sort_idxs[param_name]]
			pp_chisq[param_name] = np.array(pp_chisq[param_name])[pp_sort_idxs[param_name]]
			pp_phi[param_name] = np.array(pp_phi[param_name])[pp_sort_idxs[param_name]]
	
		n_plot_types = 2
		param_names = tuple(pp_vals.keys())
		n_params = len(param_names)
	
		chisq_style=dict(
			color='tab:blue',
			alpha=0.6,
			linestyle='-',
			linewidth=1,
			marker='.'
		)
		
		phi_style=dict(
			color='tab:orange',
			alpha=0.6,
			linestyle='-',
			linewidth=1,
			marker='.'
		)
	
		# fig info
		fig_spec = dict(
			figsize=(4*n_params, 4*n_plot_types)
		)
		
		# subfig info
		subfig_spec = dict(
			nrows=1, 
			ncols=1, 
			squeeze=False
		)
		
		# subplot info
		subplot_spec = dict(
			nrows = n_plot_types, 
			ncols = n_params,
			squeeze=False
		)
		
		
		f = plt.figure(**fig_spec)
		subfigs = f.subfigures(**subfig_spec).flatten()
		sf_axes = tuple(subfig.subplots(**subplot_spec) for subfig in subfigs)
	
		axes = sf_axes[0]
		_lgr.debug(f'{axes.shape=}')
		
		subfigs[0].suptitle('retrival result statistics vs retrieval set parameters')
				
		for i, axs in enumerate(axes.T):
			first_col_flag = i == 0
			#last_col_flag = i == (n_params-1)
			
			param_name = param_names[i]
			
			with Value(axs[0]) as ax:
				ax.plot(
					pp_vals[param_name],
					pp_chisq[param_name],
					label='chisq' if first_col_flag else None,
					**chisq_style,
				)
				[a.set_visible(False) for a in ax.get_xticklabels(which='both')]
				[a.set_visible(first_col_flag) for a in ax.get_yticklabels(which='both')]
				
				if first_col_flag:
					ax.set_ylabel('chisq (goodness of fit)')
				
			
			with Value(axs[1]) as ax:
				ax.plot(
					pp_vals[param_name],
					pp_phi[param_name],
					label='phi' if first_col_flag else None,
					**phi_style,
				)
				[a.set_visible(True) for a in ax.get_xticklabels(which='both')]
				[a.set_visible(first_col_flag) for a in ax.get_yticklabels(which='both')]
				
				if first_col_flag:
					ax.set_ylabel('phi (cost function)')
				
				# last row, add x-label
				ax.set_xlabel(param_name)
				
		subfigs[0].legend()
		
		if plot_fpath is not None:
			plot_fpath.parent.mkdir(parents=True, exist_ok=True)
			f.savefig(plot_fpath, bbox_inches='tight')
		
		return
	
	
	def comparison_plots(
			self, 
			plot_path_fmt : PathFormat = PathFormat('{retrieval_set_dir}/plots/{plot_name}.png'), 
			show_plots : bool = True,
			IGNORE_READONLY = False,
		) -> None:
		"""
		Create comparison plots between elements of the retrieval set.
		"""

		self.plot_measurement_comparison(plot_path_fmt, IGNORE_READONLY)
		
		if show_plots:
			plt.show()
	
	
	def parameter_plots(
			self,
			parameters : None | tuple[str,...] = ('latitude', 'longitude'),
			plot_path_fmt : PathFormat = PathFormat('{retrieval_set_dir}/plots/{plot_name}.png'), 
			show_plots : bool = True,
			IGNORE_READONLY = False,
		) -> None:
		"""
		Create plots of attributes of elements of the retrieval set vs the parameters of the retrieval set element. 
		"""
		
		
		self.plot_measurement_vs_parameters(
			parameters = parameters,
			plot_path_fmt = plot_path_fmt, 
			IGNORE_READONLY = IGNORE_READONLY,
		)
		
		
		self.plot_model_parameters_vs_parameters(
			parameters = parameters,
			plot_path_fmt = plot_path_fmt, 
			IGNORE_READONLY = IGNORE_READONLY,
		)
		
		self.plot_fit_stats_vs_parameters(
			parameters = parameters,
			plot_path_fmt = plot_path_fmt,
			IGNORE_READONLY = IGNORE_READONLY,
		)
		
		if show_plots:
			plt.show()
		
	
	
	


if __name__=='__main__':
	import argparse as ap
	import ltda.cfg.logs
	import archnemesis as ans
	
	log_controller = ltda.cfg.logs.LogArgController()
	
	
	parser = ap.ArgumentParser()
	parser.add_argument('paths', type=str, nargs='+', help='Path to a retrieval set directory')
	parser.add_argument('--no_show_plots', action='store_true', help='if present, will not show plots (but will still write them to disk)', default=False)
	parser.add_argument('--plot_path_fmt', type=PathFormat.from_argument, help=PathFormat.get_description(extra_keywords=('retrieval_set_dir', 'plot_name',)) + ' (default = "{retrieval_set_dir}/plots/{plot_name}.png")', default=PathFormat('{retrieval_set_dir}/plots/{plot_name}.png'))
	parser.add_argument('--redirect_path', type=redirect_file_access.Redirector.from_string, action='append', help='Creates a redirect from as string formatted as "{old_path}->{new_path}" any paths relative to {old_path} are redirected to be relative to {new_path}', default=[])
	
	log_group = log_controller.add_arguments_to_parser(parser)
	
	args = parser.parse_args(sys.argv[1:])
	_lgr.info('ARGUMENTS\n' + '\n'.join(f'\t{k} : {v}' for k,v in vars(args).items())+'\nEND ARGUMENTS')
	
	log_controller.set_from_arguments(args)
	ans.cfg.logs.set_packagewide_level(log_controller.current_level)
	#_lgr.setLevel(log_controller.current_level)
	
	
	for path in args.paths:
		_lgr.info(f'Operating on retrieval set at "{path}"')
		retrieval_set = RetrievalSet.from_dir(
			path, 
			path_redirects=tuple(args.redirect_path),
		)
	
		retrieval_set.comparison_plots(
			show_plots = not args.no_show_plots,
			plot_path_fmt = args.plot_path_fmt,
			IGNORE_READONLY=True,
		)
		
		retrieval_set.parameter_plots(
			show_plots = not args.no_show_plots,
			plot_path_fmt = args.plot_path_fmt,
			IGNORE_READONLY=True,
		)