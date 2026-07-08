
import os, os.path
from pathlib import Path
from typing import Self, Any, TYPE_CHECKING, NamedTuple
import dataclasses as dc
import contextlib
import inspect 
import copy
import shutil


import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt


import archnemesis as ans
from archnemesis.enum import (
	AtmosphericProfileTypeEnum,
	ArchNemesisFileTypeEnum,
)
from archnemesis.extensions.pathlib_extensions import PathFormat

from . import redirect_file_access

from .plotter import (
	AtmospherePlotter,
	CIAPlotter,
	LayerOpacityPlotter,
	LayerPlotter,
	MeasurementPlotter,
	OptimalEstimationPlotter,
)

import archnemesis.cfg.logs as logging
_lgr = logging.getLogger("archnemesis.Retrieval.Retrieval")
_lgr.setLevel(logging.INFO)


if TYPE_CHECKING:
	SHAPE = "A tuple of integers that defines the shape of a numpy array"


class RetrievalStoredStateWarning(RuntimeWarning):
	"""
	The stored state of the retrieval is different from the computed state when reading HDF5 file for some reason
	"""



@dc.dataclass
class RetrievedParams:
	NVAR : Any # number of retrieved profile parameterisations
	NXVAR: Any # number of parameters associated with each profile
	VARIDENT : Any # variable parameterisation ids
	VARPARAM : Any # Extra parameters for profiles (not retrieved, these are constants)
	APRPARAM : Any # apriori parameters (before retrieval)
	APRERRPARAM : Any # apriori error on parameters (before retrieval)
	RETPARAM : Any # retrieved parameters
	RETERRPARAM : Any # error on retrieved parameters

class SpectralQuantity(NamedTuple):
	spectral_point : float
	value : float
	error : float

class SpectralValue(NamedTuple):
	spectral_point : float
	value : float



def frac_diff(a : np.ndarray, b : np.ndarray) -> np.ndarray:
	"""
	Get the fractional difference between `a` and `b`
	"""
	both_zero_mask = (a==0) & (b==0)
	result = np.abs(a - b) / np.maximum(np.abs(a), np.abs(b))
	result[both_zero_mask] = 0
	return result


def is_frac_diff_within_limit(a : np.ndarray, b : np.ndarray, limit : float) -> bool:
	"""
	Test the fractional difference between `a` and `b` is less than `limit`
	"""
	return np.all(frac_diff(a,b) < limit)


@dc.dataclass(slots=True)
class RetrievalData:
	_runname : str
	Atmosphere : ans.Atmosphere_0
	Measurement : ans.Measurement_0
	Spectroscopy : ans.Spectroscopy_0
	Scatter : ans.Scatter_0
	Stellar : ans.Stellar_0
	Surface : ans.Surface_0
	CIA : ans.CIA_0
	Layer : ans.Layer_0
	Variables : ans.Variables_0
	Telluric : ans.Telluric_0
	RetrievalEngine : ans.OptimalEstimation_0 | ans.NestedSampling_0
	
	@property
	def runname(self)->str:
		return self._runname
	
	@runname.setter
	def runname(self, value:str):
		self._runname = value
		self.Atmosphere.runname = value
		self.Measurement.runname = value
		self.Spectroscopy.RUNNAME = value
		self.CIA.runname = value


@dc.dataclass(slots=True)
class Retrieval:
	"""
	Holds all the data required to specify an ArchNemesis model, plus methods to operate on that data.
	"""
	_runname : str # runname of ArchNemesis model
	
	# internal attributes are keyword-only attributes and are prefixed with an underscore ('_')
	
	_ : dc.KW_ONLY # all following attributes are keyword only attributes
	
	_data : None | RetrievalData = dc.field(default=None, repr=False, compare=True) # data of ArchNemesis model, can be None, if so will be loaded on first use.
	_working_directory : str | None = dc.field(default=None, repr=False, compare=True)
	_filetype : ArchNemesisFileTypeEnum = dc.field(default=ArchNemesisFileTypeEnum.UNDEFINED, repr=False, compare=False)
	_path_redirects : tuple[redirect_file_access.Redirector] = dc.field(default=tuple(), repr=False, compare=False)
	_metadata : dict[str,Any] = dc.field(default_factory=dict, repr=False, compare=False)
	_acceptable_stored_vs_input_apriori_err : float = dc.field(default=1E-3, repr=False, compare=False) # acceptable error on stored vs input apriori state vector
	_cached_forward_model_instance : None|ans.ForwardModel_0 = dc.field(default=None, repr=False, compare=False)
	
	def __repr__(self)->str:
		return f'Retrieval(working_directory={self.working_directory}, runname={self.runname}, filetype={self._filetype})'
	
	## Static Helpers ##
	
	@staticmethod
	def set_log_level(log_level : int) -> None:
		"""
		Sets the log level for `archnemesis` to `log_level`.
		
		E.g. `retrieval.set_log_leve(logging.DEBUG)`
		"""
		_lgr.setLevel(log_level)
	
		ans.cfg.logs.set_packagewide_level(log_level)
				
		print(f'Logging level set to: {logging.getLevelName(_lgr.level)}')
		return
	
	
	## Factories ##
	
	
	
	@classmethod
	def from_hdf5(
			cls, 
			runname : str,
			path_redirects : tuple[redirect_file_access.Redirector] = tuple(),
			lazy_load = True,
		) -> Self:
		
		instance = cls(
			runname,
			_data = None,
			_working_directory = os.path.abspath(os.getcwd()),
			_filetype = ArchNemesisFileTypeEnum.HDF5,
			_path_redirects = path_redirects,
		)
		
		if not lazy_load:
			instance.load(reload=False)
		
		return instance
	
	@classmethod
	def from_legacy(
			cls, 
			runname : str,
			path_redirects : tuple[redirect_file_access.Redirector] = tuple(),
			lazy_load=True,
		)-> Self:
		
		instance = cls(
			runname,
			_data = None,
			_working_directory = os.path.abspath(os.getcwd()),
			_filetype = ArchNemesisFileTypeEnum.LEGACY,
			_path_redirects = path_redirects,
		)
		
		if not lazy_load:
			instance.load(reload=False)
		
		return instance
	
	@classmethod
	def from_file(
			cls, 
			fpath : str,
			path_redirects : tuple[redirect_file_access.Redirector] = tuple(),
			lazy_load = True,
		) -> Self:
		"""
		Creates a retrieval instance from the specified file
		"""
		fpath = os.path.abspath(str(fpath))
		
		dirname = os.path.dirname(fpath)
		runname, ext = os.path.splitext(os.path.basename(fpath))
		
		instance = None
		if ext == '.h5':
			instance = cls(
				runname,
				_data = None,
				_working_directory = dirname,
				_filetype = ArchNemesisFileTypeEnum.HDF5,
				_path_redirects = path_redirects,
			)
			
		else:
			instance = cls(
				runname,
				_data = None,
				#_working_directory = os.path.abspath(os.getcwd()),
				_working_directory = os.path.abspath(dirname),
				_filetype = ArchNemesisFileTypeEnum.LEGACY,
				_path_redirects = path_redirects,
			)
		
		if instance is None:
			raise RuntimeError(f'Something went wrong when reading Retrieval from "{fpath}"')
		
		if not lazy_load:
			instance.load(reload=False)
		
		return instance
	
	@classmethod
	def from_runname(
			cls, 
			runname_path : str, # e.g. ".../some/dir/runname" ends in a runname not in an actual file
			path_redirects : tuple[redirect_file_access.Redirector] = tuple(),
			lazy_load = True,
		) -> Self:
		"""
		Creates a retrieval instance from the specified file
		"""	
		dirname = os.path.dirname(runname_path)
		runname = runname_path.rsplit(os.sep, maxsplit=1)[1]
		
		hdf5_ext = '.h5'
		legacy_ext = '.inp'
		
		test_hdf5_path = runname_path + hdf5_ext
		test_legacy_path = runname_path + legacy_ext
		
		instance = None
		if os.path.exists(test_hdf5_path):
			instance = cls(
				runname,
				_data = None,
				_working_directory = os.path.abspath(dirname),
				_filetype = ArchNemesisFileTypeEnum.HDF5,
				_path_redirects = path_redirects,
			)
			
		elif os.path.exists(test_legacy_path):
			instance = cls(
				runname,
				_data = None,
				#_working_directory = os.path.abspath(os.getcwd()),
				_working_directory = os.path.abspath(dirname),
				_filetype = ArchNemesisFileTypeEnum.LEGACY,
				_path_redirects = path_redirects,
			)
		else:
			raise FileNotFoundError(f'Could not find HDF5 file {test_hdf5_path} or LEGACY file {test_legacy_path}')
		
		if instance is None:
			raise RuntimeError(f'Something went wrong when reading Retrieval from "{runname_path}"')
		
		if not lazy_load:
			instance.load(reload=False)
		
		return instance
	
	@classmethod
	def from_dir(
			cls, 
			dir_path : str,
			path_redirects : tuple[redirect_file_access.Redirector] = tuple(),
			lazy_load = True
		) -> Self:
		"""
		Creates a retrieval instance from the specified directory, attempts to find the correct files, can fail and raise a FileNotFoundError
		"""
		if not os.path.exists(dir_path):
			raise FileNotFoundError(f'the directory "{dir_path}" does not exist')
		
		if not os.path.isdir(dir_path):
			raise NotADirectoryError(f'the path "{dir_path}" is not a directory')
		
		files = os.listdir(dir_path)
		input_file = None
		
		# Look for things that could be input files either <runname>.inp or <runname>.h5 files
		legacy_input_files = tuple(filter(lambda x: x.endswith('.inp'), files))
		hdf5_input_files = tuple(filter(lambda x: x.endswith('.h5'), files))
		
		if len(legacy_input_files) > 1:
			raise RuntimeError(f'When constructing Retrieval from directory "{dir_path}". Found more than one <runname>.inp file {legacy_input_files}')
		if len(hdf5_input_files) > 1:
			raise RuntimeError(f'When constructing Retrieval from directory "{dir_path}". Found more than one <runname>.h5 file {hdf5_input_files}')
		
		if len(legacy_input_files) == 0 and len(hdf5_input_files) == 0:
			raise FileNotFoundError(f'When constructing Retrieval from directory "{dir_path}". No <runname>.inp file or <runname>.h5 file was found')
		
		if len(legacy_input_files)==1 and len(hdf5_input_files) == 1:
			_lgr.warning(f'When constructing Retrieval from directory "{dir_path}". Both {legacy_input_files[0]} file AND {hdf5_input_files[0]} file were found. DEFAULTING to {hdf5_input_files[0]}')
		
		
		if input_file is None:
			if len(hdf5_input_files) == 1:
				input_file = hdf5_input_files[0]
			elif len(legacy_input_files)==1:
				input_file = legacy_input_files[0]
			else:
				raise RuntimeError(f'When constructing Retrieval from directory "{dir_path}", something went wrong determining the input file')
		else:
			_lgr.warning(f'When constructing Retrieval from directory "{dir_path}", {input_file=} has somehow been set from somewhere else')
		
		return Retrieval.from_file(os.path.abspath(os.path.join(dir_path, input_file)), path_redirects = path_redirects, lazy_load=lazy_load)
		
	
	@classmethod
	def copy(
			cls, 
			other : Self, 
		) -> Self:
		"""
		Make a complete copy of the instance
		"""
		instance = copy.deepcopy(other)
		return instance
		
		
	
	
	## Initialisation ##
	
	def load(self, reload=False) -> Self:
		if reload or (self._data is None):
			_lgr.info(f'Loading ArchNemesis data for {self}')
			self._load_data()
		else:
			_lgr.debug(f'ArchNemesis data for {self} already loaded')
		return self
	
	def _load_data(self) -> None:
		with contextlib.chdir(self.working_directory):
			if self._filetype == ArchNemesisFileTypeEnum.HDF5:
				self._load_data_hdf5()
			elif self._filetype == ArchNemesisFileTypeEnum.LEGACY:
				self._load_data_legacy()
			else:
				raise RuntimeError(f'Cannot load data, unknown filetype {self._filetype=}')
			
			self._ensure_retrieval_result_consistency()
			self._ensure_layer_angle_consistency()
	
	
	def _load_data_hdf5(self) -> None:
		with redirect_file_access.using(*self._path_redirects):
			(
				Atmosphere,
				Measurement,
				Spectroscopy,
				Scatter,
				Stellar,
				Surface,
				CIA,
				Layer,
				Variables,
				RetrievalEngine,
				Telluric,
			) = ans.read_input_files_hdf5(self.runname)
			
			self._data = RetrievalData(
				self.runname, 
				Atmosphere,
				Measurement,
				Spectroscopy.set_table_location_redirects(tuple(x.to_tuple() for x in self._path_redirects)),
				Scatter,
				Stellar,
				Surface,
				CIA,
				Layer,
				Variables,
				Telluric,
				RetrievalEngine,
			)
		
		if self._data is None:
			raise RuntimeError('Could not open RetrievalData from HDF5 file')
		
		retrieved_values_valid = True
		retrieved_params = RetrievedParams(*ans.read_retparam_hdf5(self.runname))
		
		#_lgr.debug(f'{retrieved_params=}')
		
		# Test that our retrieved parameters are actually valid
		if any(getattr(retrieved_params,field.name) is None for field in dc.fields(retrieved_params)):
			_lgr.info(f'Some retrieved parameters are not present in HDF5 file: {", ".join((field.name for field in dc.fields(retrieved_params) if getattr(retrieved_params,field.name) is None))}')
			retrieved_values_valid = False
		else:
			if (self._data.Variables.NVAR != retrieved_params.NVAR):
				_lgr.info(f'Retrieved parameters from HDF5 file does not have the same number of parameterisations as specified in the {self.runname}.apr file')
				retrieved_values_valid = False
			if retrieved_values_valid and np.any(self._data.Variables.NXVAR != retrieved_params.NXVAR):
				_lgr.info(f'Retrieved parameters from HDF5 file does not have the same number of parameters associated with each parameterised profile as specified in the {self.runname}.apr file')
				retrieved_values_valid = False
			if retrieved_values_valid and np.any(self._data.Variables.VARIDENT != retrieved_params.VARIDENT):
				_lgr.info(f'Retrieved parameters from HDF5 file does not have the same variable identification codes (VARIDENT) as specified in the {self.runname}.apr file')
				retrieved_values_valid = False
		
		if not retrieved_values_valid:
			_lgr.info('Retrieved values from HDF5 file are out-of-sync with the current Retrieval, we are not setting the Retrieval.Variables.XN (retrieved state vector).')
		else:
			# Still need to do some tests
			retrieved_state_vector_values = np.full(self._data.Variables.XN.shape, fill_value=np.nan, dtype=float)
			retrieved_error_values = np.full(self._data.Variables.XN.shape, fill_value=np.nan, dtype=float)
			
			sv_idx = 0 # state vector index
			
			# unpack retrieved apriori state vector and error
			for ivar in range(retrieved_params.NVAR):
				for i in range(retrieved_params.NXVAR[ivar]):
					retrieved_state_vector_values[sv_idx] = retrieved_params.APRPARAM[i,ivar]
					retrieved_error_values[sv_idx] = retrieved_params.APRERRPARAM[i,ivar]
					sv_idx += 1
			
			# Compare with self._data.Variables.XA
			sv_exp = np.exp(self._data.Variables.XA)
			sv = np.where(self._data.Variables.LX == 1, sv_exp, self._data.Variables.XA)
			temp = np.sqrt(np.diagonal(self._data.Variables.SA))
			ev = temp*sv
			sm = np.where(self._data.Variables.LX == 1, ev, temp)
			
			if not is_frac_diff_within_limit(sv, retrieved_state_vector_values, self._acceptable_stored_vs_input_apriori_err):
				_lgr.info(f'Apriori state vector from retrieval is not the same as from {self.runname}.apr file.')
				retrieved_values_valid = False
			if not is_frac_diff_within_limit(sm, retrieved_error_values, self._acceptable_stored_vs_input_apriori_err):
				_lgr.info(f'Apriori error (square-root diagonal of covariance matrix) from retrieval is not the same as from {self.runname}.apr file.')
				retrieved_values_valid = False
			
			
			# If retrieved parameters are valid at this point, then we can use them
			if not retrieved_values_valid:
				_lgr.info('Retrieved values from HDF5 file are out-of-sync with the current Retrieval, we are not setting the Retrieval.Variables.XN (retrieved state vector).')
			else:
				_lgr.info('Retrieved values from HDF5 file are in-sync with current retrieval. We can set the Retrieval.Variables.XN (retrieved state vector)')
				retrieved_state_vector_values = np.full(self._data.Variables.XN.shape, fill_value=np.nan, dtype=float)
				retrieved_error_values = np.full(self._data.Variables.XN.shape, fill_value=np.nan, dtype=float)
				
				sv_idx = 0 # state vector index
				
				# unpack retrieved apriori state vector and error
				for ivar in range(retrieved_params.NVAR):
					for i in range(retrieved_params.NXVAR[ivar]):
						retrieved_state_vector_values[sv_idx] = retrieved_params.RETPARAM[i,ivar]
						retrieved_error_values[sv_idx] = retrieved_params.RETERRPARAM[i,ivar]
						sv_idx += 1
			
				# Log etc. as needed
				sv_log = np.log(retrieved_state_vector_values)
				se_frac = retrieved_error_values/retrieved_state_vector_values
				
				self._data.Variables.XN[...] = np.where(self._data.Variables.LX==1, sv_log, retrieved_state_vector_values)
				np.fill_diagonal(self._data.Variables.SX, np.where(self._data.Variables.LX==1, se_frac, retrieved_error_values)**2)
		return
	
	
	def _load_data_legacy(self) -> None:
		with redirect_file_access.using(*self._path_redirects):
			(
				Atmosphere,
				Measurement,
				Spectroscopy,
				Scatter,
				Stellar,
				Surface,
				CIA,
				Layer,
				Variables,
				RetrievalEngine,
			) = ans.read_input_files(self.runname)
			Telluric = None
			
			self._data = RetrievalData(
				self.runname, 
				Atmosphere,
				Measurement,
				Spectroscopy.set_table_location_redirects(tuple(x.to_tuple() for x in self._path_redirects)),
				Scatter,
				Stellar,
				Surface,
				CIA,
				Layer,
				Variables,
				Telluric,
				RetrievalEngine,
			)
			
			if self._data is None:
				raise RuntimeError('Could not load RetrievalData from LEGACY files')
		return
	
	
	def _ensure_retrieval_result_consistency(self):
		_lgr.debug('Determine validity of loaded self.RetrievalEngine, check lengths of arrays and values of apriori state vector')
		
		# Check state vector is consistent
		retrieved_values_valid = self.RetrievalEngine.NX == self.Variables.NX
		
		if not retrieved_values_valid:
			_lgr.warning(f'Stored apriori state vector has different length than found from *.apr file, {self.RetrievalEngine.NX=} {self.Variables.NX=}')
		else:
			retrieved_values_valid &= is_frac_diff_within_limit(self.RetrievalEngine.XA, self.Variables.XA, self._acceptable_stored_vs_input_apriori_err)
		
			if not retrieved_values_valid:
				_lgr.debug(f'{self.RetrievalEngine.XA=}')
				_lgr.debug(f'{self.Variables.XA=}')
				_lgr.debug(f'{frac_diff(self.RetrievalEngine.XA, self.Variables.XA)=}')
				_lgr.warning('Stored apriori state vector has different entries than found from *.apr file.')
		
		# Check measurement vector is consistent
		retrieved_values_valid &= self.RetrievalEngine.NY == self.Measurement.NY
		
		if not retrieved_values_valid:
			_lgr.debug(f'{self.RetrievalEngine.NY=} {self.Measurement.NY=}')
			_lgr.warning(f'Stored measurement vector has different lengths in RetrievalEngine ({self.RetrievalEngine.NY}) vs Measurement ({self.Measurement.NY})')
		else:
			retrieved_values_valid *= is_frac_diff_within_limit(self.RetrievalEngine.Y, self.Measurement.Y, self._acceptable_stored_vs_input_apriori_err)
		
			if not retrieved_values_valid:
				_lgr.debug(f'{frac_diff(self.RetrievalEngine.Y, self.Measurement.Y)=}')
				_lgr.warning('Store measurement vector has different entries in RetrievalEngine vs Measurement.')
		
		# 
		if not retrieved_values_valid:
			_lgr.warning('Optimal estimation setup is out-of-sync with the current Retrieval. Removing retrieved arrays and updating apriori arrays.')
			# Remove all retrieved arrays as they are invalid for the current setup
			# self.Variables
			self.RetrievalEngine.XN = None
			self.RetrievalEngine.ST = None
			self.RetrievalEngine.KK = None
			self.RetrievalEngine.AA = None
			self.RetrievalEngine.DD = None
			# self.Measurement
			self.RetrievalEngine.YN = None
			
			# Update apriori arrays to match
			# self.Variables
			self.RetrievalEngine.XA = np.array(self.Variables.XA)
			self.RetrievalEngine.SA = np.array(self.Variables.SA)
			# self.Measurement
			self.RetrievalEngine.Y = self.Measurement.Y
			self.RetrievalEngine.SE = self.Measurement.SE
			
			
			# Update state vector size to match
			# self.Variables
			self.RetrievalEngine.NX = self.Variables.NX
			# self.Measurement
			self.RetrievalEngine.NY = self.Measurement.NY
		
		else:
			_lgr.info('Optimal estimation setup is in-sync with the current Retrieval.')
			if (
					(self.RetrievalEngine.YN is not None) 
					and (
						(not hasattr(self.RetrievalEngine, 'PHI'))
						or (not hasattr(self.RetrievalEngine, 'CHISQ'))
					)
				):
					self.RetrievalEngine.calc_phiret()
		
		
		if retrieved_values_valid and self.RetrievalEngine.YN is not None:
			# We can assume that the retrieved spectra is ok to use if we have gotten here and still have a retrieved spectra
			_lgr.info('Setting self.Measurement.SPECMOD to retrieved spectra')
			if self.Measurement.SPECMOD is None:
				self.Measurement.SPECMOD = np.full(self.Measurement.MEAS.shape, fill_value=np.nan, dtype=float)
			
			j = 0
			for i, n in enumerate(self.Measurement.NCONV):
				self.Measurement.SPECMOD[:n, i] = self.RetrievalEngine.YN[j:j+n]
				j += n
		
		return
	
	def _ensure_layer_angle_consistency(self):
		"""
		Enusre that the angle of the layers is the same as the observation zenith angle
		"""
		if self._data.Layer.LAYANG is None:
			if np.all(self._data.Measurement.EMISS_ANG < 0):
				self._data.Layer.LAYANG = 90.0
				_lgr.warning(f'Layer.LAYANG is not set. Measurements are all LIMB measurements so setting to {self._data.Layer.LAYANG}')
			elif np.all(self._data.Measurement.EMISS_ANG >= 0):
				self._data.Layer.LAYANG = 0.0
				_lgr.warning(f'Layer.LAYANG is not set. Measurements are all NADIR measurements so setting to {self._data.Layer.LAYANG}')
			else:
				self._data.Layer.LAYANG = 0.0
				_lgr.warning(f"Layer.LAYANG not set. Measurements are a combination of LIMB and NADIR. Setting to {self._data.Layer.LAYANG} but layer plot results will not reflect all measurements.")
	
	
	## Interrogators ##
	
	def has_modelled_spectra(self) -> bool:
		return self.Measurement.SPECMOD is not None
	
	def is_solar_occultation_or_limb_measurement(self) -> bool:
		return self.Measurement.TANHE is not None
	
	
	## Properties ##
	
	@property
	def runname(self) -> str:
		return self._runname
	
	@runname.setter
	def runname(self, value : str) -> None:
		self._runname = value
		if self._data is not None:
			self._data.runname = self.runname
	
	@property
	def Atmosphere(self) -> ans.Atmosphere_0:
		return self.load(reload=False)._data.Atmosphere

	@Atmosphere.setter
	def Atmosphere(self, value : ans.Atmosphere_0) -> None:
		self.load(reload=False)._data.Atmosphere = value


	@property
	def Measurement(self) -> ans.Measurement_0:
		return self.load(reload=False)._data.Measurement

	@Measurement.setter
	def Measurement(self, value : ans.Measurement_0) -> None:
		self.load(reload=False)._data.Measurement = value


	@property
	def Spectroscopy(self) -> ans.Spectroscopy_0:
		return self.load(reload=False)._data.Spectroscopy

	@Spectroscopy.setter
	def Spectroscopy(self, value : ans.Spectroscopy_0) -> None:
		# Make sure we have the same redirects in place for the new Spectroscopy_0 instance
		value.set_table_location_redirects(tuple(x.to_tuple() for x in self._path_redirects))
		self.load(reload=False)._data.Spectroscopy = value


	@property
	def Scatter(self) -> ans.Scatter_0:
		return self.load(reload=False)._data.Scatter

	@Scatter.setter
	def Scatter(self, value : ans.Scatter_0) -> None:
		self.load(reload=False)._data.Scatter = value


	@property
	def Stellar(self) -> ans.Stellar_0:
		return self.load(reload=False)._data.Stellar

	@Stellar.setter
	def Stellar(self, value : ans.Stellar_0) -> None:
		self.load(reload=False)._data.Stellar = value


	@property
	def Surface(self) -> ans.Surface_0:
		return self.load(reload=False)._data.Surface

	@Surface.setter
	def Surface(self, value : ans.Surface_0) -> None:
		self.load(reload=False)._data.Surface = value


	@property
	def CIA(self) -> ans.CIA_0:
		return self.load(reload=False)._data.CIA

	@CIA.setter
	def CIA(self, value : ans.CIA_0) -> None:
		self.load(reload=False)._data.CIA = value


	@property
	def Layer(self) -> ans.Layer_0:
		return self.load(reload=False)._data.Layer

	@Layer.setter
	def Layer(self, value : ans.Layer_0) -> None:
		self.load(reload=False)._data.Layer = value


	@property
	def Variables(self) -> ans.Variables_0:
		return self.load(reload=False)._data.Variables

	@Variables.setter
	def Variables(self, value : ans.Variables_0) -> None:
		self.load(reload=False)._data.Variables = value


	@property
	def Telluric(self) -> ans.Telluric_0:
		return self.load(reload=False)._data.Telluric

	@Telluric.setter
	def Telluric(self, value : ans.Telluric_0) -> None:
		self.load(reload=False)._data.Telluric = value


	@property
	def RetrievalEngine(self) -> ans.OptimalEstimation_0:
		return self.load(reload=False)._data.RetrievalEngine

	@RetrievalEngine.setter
	def RetrievalEngine(self, value : ans.OptimalEstimation_0) -> None:
		self.load(reload=False)._data.RetrievalEngine = value


	@property
	def measured_spectra(self) -> tuple[np.ndarray['SHAPE',float], np.ndarray['SHAPE',float], np.ndarray['SHAPE',float]]:
		return SpectralQuantity(self.Measurement.VCONV, self.Measurement.MEAS, self.Measurement.ERRMEAS)
	
	@measured_spectra.setter
	def measured_spectra(self, values_and_error : tuple[np.ndarray, np.ndarray]) -> None:
		self.Measurement.edit_MEAS(values_and_error[0])
		self.Measurement.edit_ERRMEAS(values_and_error[1])
	
	@property
	def modelled_spectra(self) -> tuple[np.ndarray['SHAPE',float], np.ndarray['SHAPE',float]]:
		return SpectralValue(self.Measurement.VCONV, self.Measurement.SPECMOD)
	
	@property
	def working_directory(self) -> str:
		return self._working_directory
	
	@working_directory.setter
	def working_directory(self, value : str) -> None:
		"""
		Set the working directory to a new value such that everything will run properly in the new 
		directory without referencing anything from the old location.
		
		NOTE: This does NOT clean up old files.
		"""
		value = os.path.abspath(str(value))
		
		if not os.path.exists(value):
			_lgr.warning(f'Desired working directory "{value}" does not exist, will create the directory.')
			os.makedirs(value)

		if not os.path.isdir(value):
			raise NotADirectoryError(f'Cannot set working directory to "{value}" as that path is not a directory')
		
		
		if not os.path.samefile(self.working_directory, value):
			# Copy supporting files as well as just setting the value
			_lgr.info(f'Working directory will be changed, copying supporting files from "{self._working_directory}" to "{value}"')
			_lgr.warning('At the moment there is no way to interrogate Retrieval.Variables for extra files required. Therefore at the moment we assume that all *.dat files are needed')
			dat_files = tuple(filter(lambda x: x.endswith('.dat'), os.listdir(self._working_directory)))
			
			supporting_files = (
				'{runname}.apr',
				*dat_files
			)
			
			for supporting_file in supporting_files:
				shutil.copyfile(
					os.path.join(self._working_directory, supporting_file.format(runname = self.runname)),
					os.path.join(value, supporting_file.format(runname = self.runname))
				)
		else:
			_lgr.warning('Working directory will not be changed as passed value is the same as the current value')
		
		self._working_directory = value
		
	@property
	def file(self) -> Path:
		if self._filetype == ArchNemesisFileTypeEnum.HDF5:
			return Path(Path(self.working_directory) / f'{self.runname}.h5')
		elif self._filetype == ArchNemesisFileTypeEnum.LEGACY:
			return Path(Path(self.working_directory) / f'{self.runname}.inp')
		else:
			raise RuntimeError(f'Cannot determine `file` of retrieval when filetype is {self._filetype}')
		
	## Private Methods ##
	
	def _get_forward_model_instance(self, IGEOM : int = None, IAV : int = None, new : bool = False):
		if new or self._cached_forward_model_instance is None:
			self._cached_forward_model_instance = ans.ForwardModel_0(
				runname = self.runname, 
				Atmosphere = self.Atmosphere,
				Surface = self.Surface,
				Measurement = self.Measurement,
				Spectroscopy = self.Spectroscopy,
				Stellar = self.Stellar,
				Scatter = self.Scatter,
				CIA = self.CIA,
				Layer = self.Layer,
				Variables = self.Variables,
				Telluric = self.Telluric,
			)
			
			with redirect_file_access.using(*self._path_redirects):
				self._cached_forward_model_instance.init_for_geometry_and_averaging_point(IGEOM, IAV)
		
		return self._cached_forward_model_instance
	
	def _set_optimal_estimation_setup_from_measurement(self):
		
		self.RetrievalEngine.NY = self.Measurement.NY
		
		if self.Measurement.Y is not None:
			self.RetrievalEngine.edit_Y(self.Measurement.Y)
		
		if self.Measurement.SPECMOD is not None:
			if self.RetrievalEngine.YN is None:
				self.RetrievalEngine.YN = np.zeros((self.RetrievalEngine.NY,),float)
			ik=0
			for igeom in range(self.Measurement.NGEOM):
				self.RetrievalEngine.YN[ik:ik+self.Measurement.NCONV[igeom]] = self.Measurement.SPECMOD[0:self.Measurement.NCONV[igeom],igeom]
	
	def _set_optimal_estimation_setup_from_variables(self, variables : None | ans.Variables_0 = None):
		if variables is None:
			variables = self.Variables
		
		self.RetrievalEngine.NX = variables.NX
		self.RetrievalEngine.edit_XA(variables.XA)
		self.RetrievalEngine.edit_XN(variables.XN)
		self.RetrievalEngine.edit_SA(variables.SA)
		self.RetrievalEngine.ST = variables.SX
	
	
	## Public Methods ##
	@contextlib.contextmanager
	def working_directory_context(self):
		with contextlib.chdir(self.working_directory):
			yield
		return
	
	def write(self, filetype : ArchNemesisFileTypeEnum = ArchNemesisFileTypeEnum.UNDEFINED) -> None:
		"""
		Write the components of this retrieval to disk
		"""
		if filetype is ArchNemesisFileTypeEnum.UNDEFINED:
			filetype = self._filetype
		
		if filetype not in (ArchNemesisFileTypeEnum.HDF5, ArchNemesisFileTypeEnum.LEGACY):
			raise ValueError(f"When writing Retrieval(runname={self.runname}) {filetype=}, must be one of ({ArchNemesisFileTypeEnum.HDF5=}, {ArchNemesisFileTypeEnum.LEGACY=})")
		
		# Ensure we have data loaded
		self.load(reload=False)
		
		# Loop through components of retrieval, find methods that begin with "write"
		components = []
		for field in dc.fields(self._data):
			if not field.name.startswith('_'):
				components.append(field.name)
		
		
		component_writers = dict()
		for component in components:
			component_obj = getattr(self, component)
			component_writers[component] = []
			
			all_methods_of_component = inspect.getmembers(component_obj, inspect.ismethod)
			
			for method_name, method in all_methods_of_component:
				if method_name.startswith('write_'):
					component_writers[component].append(method)
		
		# Make sure we are in the correct directory to write the files
		with self.working_directory_context():
			_lgr.info(f'Writing Retrieval (filetype = {self._filetype}) to directory "{os.getcwd()}" with runname {self.runname}')
		
			if filetype is ArchNemesisFileTypeEnum.HDF5:
				# only want write_hdf5 methods
				for component, component_writers in component_writers.items():
					for component_writer in component_writers:
						match (component_writer.__name__):
							case "write_hdf5" |"write_input_hdf5":
								_lgr.debug(f'Calling "{component_writer}"')
								component_writer(self.runname)
							case "write_output_hdf5":
								if self.RetrievalEngine.Y is not None: # only write output if we have output to write
									_lgr.debug(f'Calling "{component_writer}"')
									component_writer(self.runname, self.Variables)
			
				return
			
			if filetype is ArchNemesisFileTypeEnum.LEGACY:
				# only want write_hdf5 methods
				for component, component_writers in component_writers.items():
					for component_writer in component_writers:
						match (component_writer.__name__):
							case "write_hdf5" |"write_input_hdf5" | "write_output_hdf5" | "write_table_hdf5" | "write_ciatable_hdf5": # don't use HDF5 writers
								continue
							case "write_phase" | "write_solar_file": # skip these for now
								continue
							case "write_xsc" | "write_sol" | "write_hap" | "write_sur": # these need the run name
								_lgr.debug(f'Calling "{component_writer}"')
								component_writer(self.runname)
							case "write_cov" :
								component_writer(self.runname, self.Variables)
							case "write_mre" :
								component_writer(self.runname, self.Variables, self.Measurement)
							case "write_raw" :
								component_writer(self.runname, self.Variables, self.Atmosphere)
							case _:
								_lgr.debug(f'Calling "{component_writer}"')
								component_writer()
				return
			else:
				raise NotImplementedError(f'Writing to the file type "{filetype}" is not implemented yet.')

	def get_all_gas_profile_models(self):
		gas_profile_models = dict()
		
		if hasattr(self.Variables, "VARIDENT")  and self.Variables.VARIDENT is not None:
			for gas_index in range(self.Atmosphere.NVMR):
				found_model = False
				gas_id, iso_id = self.Atmosphere.ID[gas_index], self.Atmosphere.ISO[gas_index]
				
				for model_index, varident in enumerate(self.Variables.VARIDENT):
					if varident[0] == gas_id and varident[1] == iso_id:
						gas_profile_models[gas_index] = self.Variables.models[model_index]
						if gas_profile_models[gas_index].target != AtmosphericProfileTypeEnum.GAS_VOLUME_MIXING_RATIO:
							raise RuntimeError(f'{self.Variables.models[model_index]} looks like a gas vmr model, but is not.')
							
						found_model = True
				
				if not found_model:
					gas_profile_models[gas_index] = None
		
		elif hasattr(self.Variables, "models"):
			for model in self.Variables.models:
				if model.is_gas_model():
					gas_profile_models[model.gas_idx] = model
			
			for gas_index in range(self.Atmosphere.NVMR):
				if gas_index not in gas_profile_models:
					gas_profile_models[gas_index] = None
		
		else:
			raise RuntimeError('No known way to identify gas models')
			
		return gas_profile_models
	
	
	def get_all_aerosol_profile_models(self):
	
		aerosol_profile_models = dict()
		
		if hasattr(self.Variables, "VARIDENT") and self.Variables.VARIDENT is not None:
			for aerosol_index in range(self.Atmosphere.NDUST):
				found_model = False
				aerosol_id = -(aerosol_index+1)
				for model_index, varident in enumerate(self.Variables.VARIDENT):
					if varident[0] == aerosol_id:
						aerosol_profile_models[aerosol_index] = self.Variables.models[model_index]
						if aerosol_profile_models[aerosol_index].target != AtmosphericProfileTypeEnum.AEROSOL_DENSITY:
							raise RuntimeError(f'{self.Variables.models[model_index]} looks like an aerosol density model, but is not.')
						found_model = True
				
				if not found_model:
					aerosol_profile_models[aerosol_index] = None
		
		elif hasattr(self.Variables, "models"):
			for model in self.Variables.models:
				if model.is_aerosol_model():
					aerosol_profile_models[model.aerosol_species_idx] = model
			
			for aerosol_index in range(self.Atmosphere.NDUST):
				if aerosol_index not in aerosol_profile_models:
					aerosol_profile_models[aerosol_index] = None
		
		else:
			raise RuntimeError('No known way to identify aerosol models')
		
		return aerosol_profile_models
	
	
	def get_spectral_profiles(self):
		raise NotImplementedError("Profiles that modify the modelled spectra are not implemented into Retrieval yet.")
	
	def calculate_apriori_profiles(self):
		forward_model_instance = self._get_forward_model_instance()
		forward_model_instance.Variables = copy.deepcopy(forward_model_instance.Variables) # copy this or we wil alter self.Variables via reference
		forward_model_instance.Variables.XN = self.Variables.XA
		forward_model_instance.Variables.SX = self.Variables.SA
		forward_model_instance.subprofretg() # perform profile calculations
		
		
		# Only the self.Atmosphere component has profiles, so update that one
		self.Atmosphere = forward_model_instance.AtmosphereX
		return self
		
	def calculate_profiles(self):
		forward_model_instance = self._get_forward_model_instance()
		forward_model_instance.subprofretg() # perform profile calculations
		
		# Only the self.Atmosphere component has profiles, so update that one
		self.Atmosphere = forward_model_instance.AtmosphereX
		return self
	
	def calculate_layering(self):
		_lgr.debug('Calculating layering for current Retrieval, will reflect current state of atmosphere (does not check if atmosphere has been calculated)')
		if self.Layer is None:
			raise RuntimeError('Retrieval.Layer component is `None`, cannot calculate layer')
		if self.Atmosphere is None:
			raise RuntimeError('Retrieval.Atmosphere component is `None`, cannot calculate layer')
		
		self.Layer.calc_layering(
			H = self.Atmosphere.H,
			P = self.Atmosphere.P,
			T = self.Atmosphere.T,
			ID = self.Atmosphere.ID, 
			VMR = self.Atmosphere.VMR,
			DUST = self.Atmosphere.DUST,
			PARAH2 = self.Atmosphere.PARAH2,
			MOLWT = self.Atmosphere.MOLWT
		)
		
		if self._cached_forward_model_instance is not None:
			self._cached_forward_model_instance.LayerX = copy.deepcopy(self.Layer)
	
	def calculate_layer_opacity(self):
		if self.Layer.PRESS is None:
			self.calculate_layering()
		
		forward_model_instance = self._get_forward_model_instance()
		if forward_model_instance.LayerX.PRESS is None:
			self.calculate_layering()
		
		forward_model_instance.calc_path()
		forward_model_instance.calculate_layer_opacity()
		self.Spectroscopy = forward_model_instance.SpectroscopyX
		self.Layer = forward_model_instance.LayerX
	
	def guess_optimal_cores(self):
		"""
		Tries to guess how many cores are optimal for running optimal estimation
		"""
		return np.count_nonzero(self.Variables.FIX==0) + 1
	
	def run_forward_model(self, apriori=False):
		if self.has_modelled_spectra():
			_lgr.warning(f"Retrieval(runname={self.runname}) already has a modelled spectra, running the forward model will overwrite the current value")
	
		forward_model_instance = self._get_forward_model_instance()
		
		if apriori:
			print('Using APRIORI values for forward model')
			forward_model_instance.Variables = copy.deepcopy(forward_model_instance.Variables) # copy this or we wil alter self.Variables via reference
			forward_model_instance.Variables.XN = self.Variables.XA
			forward_model_instance.Variables.SX = self.Variables.SA
		
		
		# Run the forward model
		with self.working_directory_context():
			with redirect_file_access.using(*self._path_redirects):
				modelled_spectra = forward_model_instance.nemesisfm()
		
		# Only the self.Atmosphere component has profiles, so update that one
		self.Atmosphere = forward_model_instance.AtmosphereX
		
		# Update the Retrieval with computed values
		self.Layer = forward_model_instance.LayerX
		self.Spectroscopy = forward_model_instance.SpectroscopyX
		self.Measurement.edit_SPECMOD(modelled_spectra)
		
		self._set_optimal_estimation_setup_from_measurement()
		self._set_optimal_estimation_setup_from_variables(forward_model_instance.Variables)
		
		return self
	
	def run_optimal_estimation(
			self, 
			n_iter : None | int = None, 
			n_cores : None | int = None, 
			write_itr : bool = False,
			restart = True,
			do_write : bool = True
	) -> Self:
		
		if restart:
			# Reset the current state vector to the apriori state vector
			self.Variables.XN[...] = self.Variables.XA
		
		n_iter = self.RetrievalEngine.NITER if n_iter is None else n_iter
		n_cores = self.guess_optimal_cores() if n_cores is None else n_cores
		
		if n_iter < 0: # If we run the forward model, return early
			with self.working_directory_context():
				_lgr.info(f'Running forward model in {self._working_directory} as {n_iter=} is less than 0')
				self.run_forward_model()
				if do_write:
					self.write()
				return self
			
		
		with self.working_directory_context():
			phi_history = None
			chisq_history = None
			
			_lgr.info(f'Running optimal estimation in {self._working_directory}')
			OptimalEstimationResult, ForwardModel, phi_history, chisq_history = ans.coreretOE(
				self.runname,
				self.Variables,
				self.Measurement,
				self.Atmosphere,
				self.Spectroscopy,
				self.Scatter,
				self.Stellar,
				self.Surface,
				self.CIA,
				self.Layer,
				self.Telluric,
				NITER = n_iter,
				PHILIMIT = self.RetrievalEngine.PHILIMIT,
				NCores = n_cores,
				nemesisSO = False,
				write_itr = write_itr,
				return_forward_model = True,
				return_phi_and_chisq_history=True,
			)
			
			if not os.path.exists('phi_chisq.txt') and (phi_history is not None and chisq_history is not None):
				with open('phi_chisq.txt', 'w') as f:
					w_iter = max(4, int(np.ceil(np.log10(OptimalEstimationResult.NITER))))
					fmt = f'{{:0{w_iter}}} | {{:09.3E}} | {{:09.3E}}\n'
					f.write(
						'iter' + ('' if w_iter <= 4 else ' '*(w_iter-4))
						+' | phi      '
						+' | chisq    \n'
					)
					for i,(p,c) in enumerate(zip(phi_history, chisq_history)):
						f.write(fmt.format(i, p, c))
			
		if n_cores is not None:
			OptimalEstimationResult.NCORES = n_cores
		
		# Update the retrieval with the result
		self.RetrievalEngine = copy.deepcopy(OptimalEstimationResult)
		
		modelled_spectra = np.zeros_like(self.Measurement.MEAS)
		ix = 0
		for i in range(self.Measurement.NGEOM):
			modelled_spectra[0:self.Measurement.NCONV[i],i] = OptimalEstimationResult.YN[ix:ix+self.Measurement.NCONV[i]]
			ix += self.Measurement.NCONV[i]
		
		self.Measurement.edit_SPECMOD(modelled_spectra)
		
		self.Layer = ForwardModel.LayerX
		
		self.calculate_profiles()
		
		if do_write:
			self.write()
	
	def run_nested_sampling(
			self,
			n_cores : None | int = None,
	) -> Self:
		raise NotImplementedError('Nested sampling is not implemented with Retrieval yet.')
	
	def get_model_parameter_table(self):
		return self.Variables.model_parameters_as_string()
	
	def interpolate_to_measured_spectra_grid(self, spectral_grid : np.ndarray, values : np.ndarray) -> None:
		original_spec_grid = self.Measurement.VCONV
		original_spec_grid_ndim = original_spec_grid.ndim
		if original_spec_grid_ndim == 1:
			original_spec_grid = original_spec_grid[:,None]
		
		if spectral_grid.ndim == 1:
			spectral_grid = spectral_grid[:, None]
		
		if values.ndim == 1:
			values = values[:,None]
		
		interp_values = np.zeros_like(original_spec_grid)
		for j in range(original_spec_grid.shape[1]):
			interp_values[:,j] = np.interp(original_spec_grid[:,j], spectral_grid[:,j], values[:,j])
		
		if original_spec_grid_ndim == 1:
			interp_values = interp_values.flatten()

		return interp_values
	
	def plot(
			self, 
			plot_path_fmt : None | PathFormat = PathFormat('{retrieval_dir}/{plot_name}.png'),
			plot_name_suffix : str = '',
		)  -> tuple[list[str], list[mpl.figure.Figure], list[tuple[np.ndarray[mpl.axes.Axes]]]]:
		"""
		Plots all possible plots for a Retrieval.
		
		## ARGUMENTS ##
	
		plot_path_fmt : None | PathFormat = PathFormat('{retrieval_dir}/{plot_name}.png')
			Format to use when getting path to plot, field names are:
				'retrieval_dir' - the working directory of the retrieval
				'plot_name' - the name of the plot
		
		plot_name_suffix : str = ''
			A suffix to append to the name of the plot, will be applied before path format.
		"""
		names, figs, axes = [], [], []
		
		
		plotter_classes = (
			AtmospherePlotter,
			MeasurementPlotter,
			LayerPlotter,
			CIAPlotter,
			LayerOpacityPlotter,
			OptimalEstimationPlotter,
		)
		
		for plotter_class in plotter_classes:
			plotter = plotter_class(self)
			if plotter.can_plot():
				name, f, ax = plotter.plot()
				names.append(name)
				figs.append(f)
				axes.append(ax)
			else:
				_lgr.warning(f'Plot not created by {plotter_class.__name__} are required components are not present on retrieval.')
		
		
		self.Variables.plot_model_parameters(self.working_directory, False)
		
		if plot_path_fmt is not None:
			with open(os.path.join(self.working_directory, 'model_parameters.tbl'), 'w') as f:
				f.write(self.Variables.model_parameters_as_string()+'\n')
				
			for name, fig in zip(names, figs):
				plot_name = name+plot_name_suffix
				fpath = plot_path_fmt.format(self.file, retrieval_dir = self.working_directory, plot_name=plot_name)
				_lgr.info(f'Saving figure "{plot_name}" to "{fpath}"')
				fpath.parent.mkdir(parents=True, exist_ok=True)
				fig.savefig(fpath, bbox_inches='tight')
			
		
		return names, figs, axes






if __name__=='__main__':
	import sys
	import argparse as ap
	import archnemesis.cfg.logs
	#import ltda.cfg.logs
	
	#log_controller = ltda.cfg.logs.LogArgController()
	
	parser = ap.ArgumentParser()
	parser.add_argument('paths', type=str, nargs='+', help='Path to a file (<runname>.inp or <runname>.h5 file) or directory (will try and guess runname)')
	parser.add_argument('--dir', type=str, help='Directory to look for input file in, if given will assume `path` should be a file NOT a directory (input file must be relative to this dir)', default=None)
	parser.add_argument(
		'-o', 
		'--operation', 
		type=str,
		nargs='+',
		choices = (
			'do_not_even_load',
			'load_only',
			'plot',
			'plot_apriori',
			'run_optimal_estimation',
			'continue_optimal_estimation',
		),
		help='Operation to perform on the retrieval (default="plot")',
		default=['plot']
	)
	parser.add_argument('-n', '--n_cores', type=int, help='Number of cores to run parallelised processes on (default is to use the number specified in the input files)', default=None)
	parser.add_argument('--iter', type=int, help='Number of iterations to perform if running optimal estimation (default is to use number specified in the input files)', default=None)
	parser.add_argument('--write_itr', action='store_true', help='if running optimal estimation, should we write a *.itr file?', default=False)
	parser.add_argument('--no_show_plots', action='store_true', help='if present, will not show plots (but will still write them to disk)', default=False)
	parser.add_argument('--plot_path_fmt', type=PathFormat.from_argument, help=PathFormat.get_description(extra_keywords=('retrieval_dir', 'plot_name',)) + ' (default = "{retrieval_dir}/{plot_name}.png")', default=PathFormat('{retrieval_dir}/{plot_name}.png'))
	parser.add_argument('--redirect_path', type=redirect_file_access.Redirector.from_string, action='append', help='Creates a redirect from as string formatted as "{old_path}->{new_path}" any paths relative to {old_path} are redirected to be relative to {new_path}', default=[])
	
	#log_group = log_controller.add_arguments_to_parser(parser)
	
	args = parser.parse_args(sys.argv[1:])
	_lgr.info('ARGUMENTS\n' + '\n'.join(f'\t{k} : {v}' for k,v in vars(args).items())+'\nEND ARGUMENTS')
	
	#log_controller.set_from_arguments(args)
	#Retrieval.set_log_level(log_controller.current_level)
		
	
	if 'do_not_even_load' in args.operation:
		_lgr.info('Not even loading retrievals, just exiting...')
		sys.exit()
	
	with redirect_file_access.using(*args.redirect_path):
		
		for path in args.paths:
		
			if args.dir is None:
				if os.path.isdir(path):
					_lgr.info(f'Loading retrieval from directory "{path}"')
					retrieval = Retrieval.from_dir(path)
				else:
					_lgr.info(f'Loading retrieval from file "{path}"')
					retrieval = Retrieval.from_file(path)
			else:
				_lgr.info(f'Loading retrieval from file "{os.path.join(args.dir, path)}"')
				retrieval = Retrieval.from_file(os.path.join(args.dir, path))
			
			
			_lgr.info('Performing operations on retrieval...')
			for op in args.operation:
				_lgr.info(f'OPERATION: {op}')
				
				if op == 'load_only':
					pass
				elif op == "plot":
					retrieval.calculate_profiles() # must do this so we are plotting the "real" state of the retrieval
					retrieval.calculate_layering()
					retrieval.calculate_layer_opacity()
					#retrieval.plot('WORKING_DIRECTORY')
					retrieval.plot(args.plot_path_fmt)
					if not args.no_show_plots:
						plt.show()
					plt.close('all')
				
				elif op == "plot_apriori":
					retrieval.calculate_apriori_profiles() # must do this so we are plotting the "real" state of the retrieval
					retrieval.calculate_layering()
					retrieval.calculate_layer_opacity()
					retrieval.plot(args.plot_path_fmt, '_apriori')
					if not args.no_show_plots:
						plt.show()
					plt.close('all')
				
				elif op == "run_optimal_estimation":
					retrieval.run_optimal_estimation(args.iter, args.n_cores, args.write_itr)
					retrieval.write()
				
				elif op == "continue_optimal_estimation":
					retrieval.run_optimal_estimation(args.iter, args.n_cores, args.write_itr, restart=False)
					retrieval.write()
				
				else:
					raise RuntimeError(f'Unknown operation "{op}" for Retrieval.')








