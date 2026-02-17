
import os
from pathlib import Path
import dataclasses as dc
from typing import NamedTuple, Type, Annotated, Callable, Literal, Any, Iterable
import textwrap 

import numpy as np
import h5py

from archnemesis.helpers import h5py_helper
from archnemesis.database.datatypes.gas_descriptor import RadtranGasDescriptor

from archnemesis.enums import AmbientGas

from archnemesis.database.offline.structured_qualifiers import IDNumber, Quantity
from archnemesis.database.offline.record_format import RecordFormat
from archnemesis.database.offline.table_format import TableFormat

# Logging
import logging
_lgr = logging.getLogger(__name__)
#_lgr.setLevel(logging.INFO)
_lgr.setLevel(logging.DEBUG)

class VirtualSourceInfo(NamedTuple):
	src_name : str
	src_file : str
	src_grp : str	

class VirtualSourceDescriptor(NamedTuple):
	src_file : str
	src_dset_path : str
	dest_dset_path : str
	src_shape : np.ndarray
	dest_offset : np.ndarray

class VirtualDestinationInfo:
	def __init__(
			self,
			expected_ndim : int = 1,
			end_offset : np.ndarray = None,
			dtype : Type = None,
			default : None | Any = None,
		):
		
		self.expected_ndim = expected_ndim
		self.end_offset = end_offset if end_offset is not None else np.zeros((self.expected_ndim,), dtype=int)
		self._dtype = dtype
		self.default = default
	
	@property
	def dtype(self):
		return self._dtype
	
	@dtype.setter
	def dtype(self, v : Type):
		if self._dtype is None:
			self._dtype = v
		elif v == self._dtype:
			pass
		else:
			raise ValueError(f'Trying to set dtype to {v} but it is already set to {self._dtype}')
	
	def __repr__(self):
		return f'VirtualDestinationInfo(expected_ndim={self.expected_ndim}, end_offset={self.end_offset}, dtype={self.dtype}, default={self.default})'


@dc.dataclass
class LineBroadenerHolder:
	name : str
	gamma_amb : None | np.ndarray = None
	n_amb : None | np.ndarray = None
	delta_amb : None | np.ndarray = None
	
	def __post_init__(self):
		if all(x is None for x in (self.gamma_amb, self.n_amb, self.delta_amb)):
			raise ValueError('LineBroadenerHolder cannot be instantiated with all ("gamma_amb", "n_amb", "delta_amb") = `None`')
		
		shape = None
		for x in (self.gamma_amb, self.n_amb, self.delta_amb):
			if x is None:
				continue
			if shape is None:
				shape = x.shape
			assert len(shape) == len(x.shape), \
				'LineBroadenerHolder must have same number of dimensions for "gamma_amb", "n_amb", "delta_amb" if they are not None'
			
			assert all(s1==s2 for s1,s2 in zip(shape,x.shape)), \
				'LineBroadenerHolder must have same shape for "gamma_amb", "n_amb", "delta_amb" if they are not None'
		
		if self.gamma_amb is None:
			self.gamma_amb = np.ones(shape, dtype=float)
		
		if self.n_amb is None:
			self.n_amb = np.zeros(shape, dtype=float)
			
		if self.delta_amb is None:
			self.delta_amb = np.zeros(shape, dtype=float)

@dc.dataclass
class LineDataHolder:
	# source information
	name : str
	description : str
	
	# spectral line data
	mol_id : np.ndarray
	local_iso_id : np.ndarray
	nu : np.ndarray
	sw : np.ndarray
	a : np.ndarray
	elower : np.ndarray
	
	# self broadening
	gamma_self : None | np.ndarray = None
	n_self : None | np.ndarray = None
	
	# foreign broadening
	broadeners : Iterable[LineBroadenerHolder] = tuple()
	
	_rt_gas_descs : None | tuple = None
	
	def __post_init__(self):
		
		if self.gamma_self is None:
			self.gamma_self = np.ones_like(self.nu, dtype=float)
		
		if self.n_self is None:
			self.n_self = np.zeros_like(self.nu, dtype=float)
		
		
		
		
	
	@property
	def rt_gas_descs(self):
		if self._rt_gas_descs is None:
			u_ids = np.unique(np.array([self.mol_id, self.local_iso_id], dtype=int), axis=1)
			print(f'DEBUG : {u_ids.shape=}')
			self._rt_gas_descs = tuple(RadtranGasDescriptor(int(gas_id), int(iso_id)) for gas_id, iso_id in u_ids.T)
		return self._rt_gas_descs


class LineDataRecordFormat(RecordFormat):
	mol_id       : Annotated[int,   IDNumber('RADTRAN ID of molecule')]
	local_iso_id : Annotated[int,   IDNumber('Isotope ID in context of parent molecule')]
	nu           : Annotated[float, Quantity('Wavenumber of line', 'cm^{-1}')]
	sw           : Annotated[float, Quantity('Line intensity at T = 296 K', 'cm^{-1}/(molec.cm^{-2})')]
	a            : Annotated[float, Quantity('Einstein A-coefficient', 's^{-1}')]
	elower       : Annotated[float, Quantity('Lower-state energy', 'cm^{-1}')]
	gamma_self   : Annotated[float, Quantity('Self-broadened HWHM at 1 atm pressure and 296 K', 'cm{^-1} atm^{-1}')]
	n_self       : Annotated[float, Quantity('Temperature exponent for the self-broadened HWHM', 'NUMBER')]

class LineBroadenerRecordFormat(RecordFormat):
	gamma_amb    : Annotated[float, Quantity('Ambient gas broadened Lorentzian half-width at half-maximum at p = 1 atm and T = 296 K', 'cm{^-1} atm^{-1}')]
	n_amb        : Annotated[float, Quantity('Temperature exponent for the ambieng gas broadened HWHM', 'NUMBER')]
	delta_amb    : Annotated[float, Quantity('Pressure shift induced by ambient gas, referred to p=1 atm', 'cm^{-1} atm^{-1}')]

class LineBroadenerTableFormat(TableFormat):
	record_format : type[RecordFormat] = LineBroadenerRecordFormat
	
	def to_hdf5(self, grp : h5py.Group, extend : None | Literal['stack'] | int = None):
		for i, name in enumerate(self.__slots__):
			#print(f'LineDataTableFormat.to_hdf5(...) {grp=} {i=} {name=} {len(getattr(self, name))=}')
			h5py_helper.ensure_dataset(
				grp, 
				name, 
				attrs=self.record_format.metadata(name).as_dict(),
				extend = extend,
				data=getattr(self, name), dtype=self.record_format.type(name), maxshape=(None,)
			)

class LineDataTableFormat(TableFormat):
	record_format : type[RecordFormat] = LineDataRecordFormat
	
	def to_hdf5(self, grp : h5py.Group, extend : None | Literal['stack'] | int = None):
		for i, name in enumerate(self.__slots__):
			#print(f'LineDataTableFormat.to_hdf5(...) {grp=} {i=} {name=} {len(getattr(self, name))=}')
			h5py_helper.ensure_dataset(
				grp, 
				name, 
				attrs=self.record_format.metadata(name).as_dict(),
				extend = extend,
				data=getattr(self, name), dtype=self.record_format.type(name), maxshape=(None,)
			)
	
	def update_hdf5(self, grp : h5py.Group, wave_attr : str = 'nu', min_wave_delta_frac = 1E-4):
	
		dsets = {}
		for i, name in enumerate(self.__slots__):
			dsets[name] = h5py_helper.get_dataset(grp, name, defaults=dict(shape=(0,), dtype=self.record_format.type(name), maxshape=(None,)))
		
		# find duplicate entries, for now assume that a line is duplicated if the fractional difference in wavelength is less than `min_wave_delta_frac`
		mask = np.zeros_like(getattr(self, self.__slots__[0]), dtype=bool)
		wave_values = getattr(self, wave_attr)
		
		print('DEBUG : Updating HDF5, therefore must check to see if data is already present.')
		dset_size = dsets[wave_attr].size
		for i, wave_dset_value in enumerate(dsets[wave_attr]):
			if i%1000 == 0:
				print(f'DEBUG : Updating HDF5 testing if data already present {i}/{dset_size} [{100*i/dset_size:6.2f} %]')
			wave_delta_frac = np.abs((wave_dset_value - wave_values)/wave_values)
			mask |= wave_delta_frac < min_wave_delta_frac # exclude entries that have small differences in wavelength
		
		mask = ~mask # negate mask so it now selects things we want to include (instead of exclude)
		n_old = dsets[wave_attr].size # old number of entries in dataset
		n_new = np.count_nonzero(mask) # number of entries to add to dataset
		print(f'DEBUG : Updating HDF5 {n_old=} {n_new=} {n_old+n_new=}')
		
		for i, name in enumerate(self.__slots__):
			#print(f'LineDataTableFormat.to_hdf5(...) {grp=} {i=} {name=} {len(getattr(self, name))=}')
			dsets[name].resize(n_old+n_new, axis=0)
			dsets[name][n_old:n_old+n_new] = getattr(self,name)[mask]
			
			for k, v in self.record_format.metadata(name).as_dict().items():
				dsets[name].attrs[k] = v






class AnsLineDataFile:
	def __init__(
			self, 
			path : None | Path = None,
			combined_isotope_group_name : str = '0'
	):
		self.path = path
		self.combined_isotope_group_name = combined_isotope_group_name
		self.default_broadening_values = {
			'gamma_amb' : 1.0,
			'n_amb' : 0.0,
			'delta_amb' : 0.0,
		}


	def repack(self):
		"""
		HDF5 does not reduce size when deleting datasets without using `h5repack` utility, however
		`h5py` does not have this capability so we just copy everything to a new file.
		"""
		try:
			temp_file = self.path.with_stem('~'+self.path.stem)
			
			with h5py.File(self.path, 'r') as f:
				with h5py.File(temp_file, 'a') as g:
					for v in f.values():
						f.copy(v, g)
			
			os.replace(temp_file, self.path)
		finally:
			if temp_file.exists():
				os.remove(temp_file)

	def dump(self):
		class HDF5Printer:
			def __init__(self, mode : Literal['indent', 'full_paths'] = 'indent'):
				self.mode = mode
				self.indent_1 = ' |  '
				self.indent_2 = ' |- '
				self.indent_3 = '    '
			
			def __call__(self, name_tail : str, item : h5py.Group | h5py.Dataset):
				if self.mode == 'indent':
					level = name_tail.count('/')
					name_last = name_tail.rsplit('/', 1)[-1]
					item_type = 'Group' if isinstance(item, h5py.Group) else f'Dataset[{item.shape}, {item.dtype}] = \n{textwrap.indent(str(item[tuple()]), (self.indent_1*(level+1)+self.indent_3))}\n{self.indent_1*(level+1)}'
					print(f'{self.indent_1*level}{self.indent_2}{name_last} : {item_type}')
				
				elif self.mode == 'full_paths':
					item_type = 'Group' if isinstance(item, h5py.Group) else f'Dataset[{item.shape}, {item.dtype}] = \n{textwrap.indent(str(item[tuple()]), ' '*(len(name_tail)+6))}'
					print(f'{name_tail} : {item_type}')
				
				else:
					raise ValueError(f'Unknown mode {self.mode=}')
		
		with h5py.File(self.path, 'r') as f:
			print(f'HDF5 File "{f.file.filename}" elements of group "{f.name}"')
			f.visititems(HDF5Printer())
		
		return

	def validate_line_data_group(self, g : h5py.Group):
		class EachChildDatasetHasSameShapeVisitor:
			def __init__(self):
				self.test_shape = None
			
			def __call__(self, name_tail : str, item : h5py.Group | h5py.Dataset):
				if not isinstance(item, h5py.Dataset): # Only operate upon datasets
					return 
				if self.test_shape is None:
					self.test_shape = item.shape
				
				assert len(self.test_shape) == len(item.shape), \
					f'Dataset "{item.name}" in "{item.file.filename}" does not have the same number of dimensions as other datasets in group "{item.name[:len(name_tail)]}". Expected ndim {len(self.test_shape)}, got ndim {len(item.shape)}.'
				
				assert all(s1==s2 for s1,s2 in zip(self.test_shape, item.shape)), \
					f'Dataset "{item.name}" in "{item.file.filename}" does not have the same shape as other datasets in group "{item.name[:len(name_tail)]}". Expected shape {self.test_shape}, got shape {item.shape}.'
		
		for mol_grp_name, mol_grp in g.items():
			if isinstance(mol_grp, h5py.Dataset):
				raise TypeError(f'Item "{mol_grp.name}" in "{g.file.filename}" is a dataset. Expected that all direct children of "line_data" group "{g.name}" are groups not datasets.')
			
			for iso_grp_name, iso_grp in mol_grp.items():
				if isinstance(iso_grp, h5py.Dataset):
					raise TypeError(f'Item "{iso_grp.name}" in "{g.file.filename}" is a dataset. Expected that all direct children of "molecule_group" group "{mol_grp.name}" are groups not datasets.')
				
				iso_grp.visititems(EachChildDatasetHasSameShapeVisitor())
		
		print(f'DEBUG : Validation for "{g.name}" in "{g.file.filename}" succeeded')
		

	def get_sources(
		self,
		s_grp : h5py.Group,
	) -> list[VirtualSourceInfo,...]:
	
		sources = []
			
		for x_name in s_grp.keys(): # loop over 'X' in /sources/X/
			x_dset = None
			x_grp = None
			x_item = s_grp[x_name]
			if isinstance(x_item, h5py.Group):
				x_grp = x_item
			elif isinstance(x_item, h5py.Dataset):
				x_dset = x_item
			else:
				raise ValueError(f'Expected h5py.Group or h5py.Dataset for entries of {s_grp.name} in {s_grp.file}, but entry {x_name} has type {type(x_item)}.')
		
			if x_grp is not None: # handle case where '/sources/X' entry is a group, and therefore should have a 'line_data' sub-group inside it.
				if 'line_data' not in x_grp.keys():
					print(f"WARNING : Source group '{x_grp.name}' in '{x_grp.file}' should have a 'line_data' sub-group. This one has entries {tuple(x_grp.keys())}, therefore not using as a source.")
					continue
				sources.append(VirtualSourceInfo(x_grp.name, '.', x_grp.name+'/line_data'))
				
			
			if x_dset is not None: # handle case where '/sources/X' entry is a dataset, therefore should have either a single string or two strings that specify an external HDF5 [file,group] pair
				external_file = None
				external_group = '/line_data'
				if len(x_dset.shape) == 0: # dataset is a scalar, so we only have the filename
					external_file = str(x_dset.astype('T')[tuple()])
				elif len(x_dset.shape) == 1 and x_dset.shape[0] == 2: #  dataset is a pair, so we have the filename and the group
					external_file, external_group = (str(x) for x in x_dset.astype('T')[:])
				else:
					raise ValueError(f'Dataset "{x_dset.name}" in "{x_dset.file}" should either be a scalar string, or a string array of shape (2,), but has dtype={x_dset.dtype} shape={x_dset.shape}')
				
				print(f'DEBUG : {external_file=} {external_group=}')
				
				sources.append(VirtualSourceInfo(x_dset.name, external_file, external_group))
		return sources


	def create_combined_isotope_groups(self, ld_grp : h5py.Group):
		
		class CombineIsotopeGroupsVisitor:
			def __init__(self, combined_isotope_group_name):
				self.combined_isotope_group_name = combined_isotope_group_name
				self.v_dest_path_map = dict()
				self.v_dest_end_offset = None
				self.last_iso_id_visited = None
			
			def __call__(self, name_tail, item):
				if not isinstance(item, h5py.Dataset):
					return
				if name_tail.startswith(self.combined_isotope_group_name):
					return
				
				v_dest_path = '/'.join((self.combined_isotope_group_name, name_tail.split('/',1)[1]))
				v_dest_src_list = self.v_dest_path_map.setdefault(v_dest_path, [])
				
				changed_iso_id = False
				if (self.last_iso_id_visited is None) or (not name_tail.startswith(self.last_iso_id_visited)):
					self.last_iso_id_visited = name_tail.split('/',1)[0]
					changed_iso_id = True
				
				v_shape = np.array(item.shape, dtype=int)
				if self.v_dest_end_offset is None:
					self.v_dest_end_offset = np.zeros_like(v_shape, dtype=int)
				
				if changed_iso_id:
					self.v_dest_end_offset += v_shape
				
				v_dest_src_list.append((item.name, v_shape, self.v_dest_end_offset - v_shape, item.dtype))
			
			
			def to_combined_iso_group(self, mol_grp : h5py.Group):
				
				for v_dest_path, v_dest_src_list in self.v_dest_path_map.items():
				
					if v_dest_path in mol_grp:
						del mol_grp[v_dest_path]

					layout = h5py.VirtualLayout(
						shape=tuple(int(s) for s in self.v_dest_end_offset), 
						dtype = np.result_type(*(x[3] for x in v_dest_src_list)),
						maxshape = tuple(None for s in self.v_dest_end_offset)
					)
					
					for v_src_path, v_shape, v_offset, dtype in v_dest_src_list:
					
						src_slices = tuple(slice(0,int(s)) for s in v_shape)
						dest_slices = tuple(slice(int(x),int(x+s)) for x,s in zip(v_offset, v_shape))
					
						vsource = h5py.VirtualSource(
							'.',
							v_src_path,
							shape=tuple(int(s) for s in v_shape),
							dtype = dtype,
							maxshape = tuple(None for s in v_shape)
						)
						
						layout[*dest_slices] = vsource[*src_slices]
							
					
					mol_grp.create_virtual_dataset(v_dest_path, layout, fillvalue=None)
		
		
		for mol_grp_name, mol_grp in ld_grp.items():
			cigv = CombineIsotopeGroupsVisitor(self.combined_isotope_group_name)
			
			mol_grp.visititems(cigv)
			
			cigv.to_combined_iso_group(mol_grp)
		


	def update_from_sources(self):
		
		with h5py.File(self.path, 'a') as f:

			ld_grp = h5py_helper.ensure_grp(f, 
				'line_data', 
				attrs = dict(
					description = "Contains nested tables of line data for each molecule/isotopologue. Some or all entries may be VIRTUAL sources which reference one or more datasets in '/sources/X/line_data'.",
				)
			)

			s_grp = h5py_helper.ensure_grp(f,
				'sources',
				attrs = dict(
					description = "Contains nested tables of line data for molecule/isotopologues from a specified sources. Entries ('/sources/X' would be the entry for source 'X') are either groups, or datasets. A group entry must have a '/sources/X/line_data' sub-group that has the same format as '/line_data'. A dataset entry must contain a filename (either relative or absolute) or a {filename,group} pair that identifies a group (if no group is specified, the group '/line_data' is assumed) within an external HDF5 file. The external group must have the same format as the '/line_data' group of this file",
				)
			)
			
			sources = self.get_sources(s_grp)
					
			self.update_virtual_datasets(ld_grp, sources)
		
			#self.create_combined_isotope_groups(ld_grp)
			self.validate_line_data_group(ld_grp)


	def update_virtual_datasets(
			self,
			ld_grp : h5py.Group,
			sources : list[VirtualSourceInfo,...],
	):
		"""
		Update the contents of "/line_data" from all virtual sources. Only updates virtual datasets, does not update
		non-virtual datasets. Will create virtual datasets for sources that have entries that do not exist in "/line_data"
		"""
		v_iso_end_offsets = dict()
		v_iso_end_updated_once_per_source = dict()
		v_dset_dest_info_map = dict()
		v_iso_dset_sources = dict()

		for src_name, x_file_name, xld_grp_path in sources:
			#print(f'DEBUG : {x_file_name=} {xld_grp_path=} {ld_grp.file=} {ld_grp.file.filename=}')
			x_file_hdl = None
			
			try:
				x_file_hdl = ld_grp if x_file_name == '.' else h5py.File(Path(ld_grp.file.filename).parent / x_file_name)
				
				if xld_grp_path not in x_file_hdl:
					raise KeyError(f'No group "{xld_grp_path}" in file "{x_file_hdl.file}".')
				
				xld_grp = x_file_hdl[xld_grp_path]
				
				self.validate_line_data_group(xld_grp)
				
				# Loop over molecule and isotopologues
				for xmol_grp_name in xld_grp.keys():
					#print(f'DEBUG : AnsLineDataFile.update_virtual_datasets(...) {xmol_grp_name=}')
					xmol_grp = xld_grp[xmol_grp_name]
					
					assert isinstance(xmol_grp, h5py.Group), f'Expected "molecule_name" group at "{xmol_grp.name}" in file "{xmol_grp.file}".'
					
					for xiso_grp_name in xmol_grp.keys():
						#print(f'DEBUG : AnsLineDataFile.update_virtual_datasets(...) {xiso_grp_name=}')
						xiso_grp = xmol_grp[xiso_grp_name]
						assert isinstance(xmol_grp, h5py.Group), f'Expected "isotope_id" group at "{xiso_grp.name}" in file "{xiso_grp.file}".'
						
						
						# Each isotopologue should have N_iso entries in each of its child datasets (that includes datasets that are children of sub-groups).
						# When creating the virtual datasets for isotopologues, we are putting all the "source" datasets together, therefore must track how
						# far through the virual dataset for an isotopologue we are. As a specific isotopologue should have N_iso entries in all of its
						# datasets we can do that at the isotopologue level.
						#
						# This simplifies things as some datasets (e.g. different gasses in "broadeners") may or may not exist for a specific isotope,
						# however, as we know that all datasets in a specific isotopologue group have N_iso entries, we can keep track of how many items
						# any missing datasets *should* have so the ones we *do* have have are aligned correctly in the virtual dataset, and missing values
						# take on a default value.
						
						v_iso_dest_path = f'{xmol_grp_name}/{xiso_grp_name}'
						v_iso_end_updated_once_per_source[v_iso_dest_path] = False
						
						v_iso_dset_sources.setdefault(v_iso_dest_path, dict())
						
						def isotope_dataset_items_visitor(name_tail : str, xdset : h5py.Group | h5py.Dataset):
							if not isinstance(xdset, h5py.Dataset): # Do not bother with groups, only datasets
								return
							elif name_tail == self.combined_isotope_group_name:
								print(f'WARNING : External source for virtual datasets should not include a "combined isotope" group {self.combined_isotope_group_name=}. However, "{xmol_grp.name}" in file "{xmol_grp.file}" includes this group, so we will skip it.')
								return 
							
							v_iso_dset_dest_path = f"{xmol_grp_name}/{xiso_grp_name}/{name_tail}"
							v_iso_dset_source_list = v_iso_dset_sources[v_iso_dest_path].setdefault(v_iso_dset_dest_path,[])
							v_shape = np.array(xdset.shape, dtype=int)
							
							v_iso_end_offset = v_iso_end_offsets.setdefault(v_iso_dest_path,np.zeros_like(v_shape, dtype=int))
							if not v_iso_end_updated_once_per_source[v_iso_dest_path]:
								v_iso_end_offset += v_shape
								v_iso_end_updated_once_per_source[v_iso_dest_path] = True
							
							v_iso_dset_source_list.append(
								VirtualSourceDescriptor(
									src_file = x_file_name,
									src_dset_path = xdset.name,
									dest_dset_path = v_iso_dset_dest_path,
									src_shape = v_shape,
									dest_offset = (v_iso_end_offset - v_shape),
								)
							)
							
							v_dset_dest_info = v_dset_dest_info_map.setdefault(v_iso_dset_dest_path, VirtualDestinationInfo())
							v_dset_dest_info.end_offset[...] = v_iso_end_offset
							v_dset_dest_info.dtype = xdset.dtype
							v_dset_dest_info.default = self.default_broadening_values.get(xdset.name.rsplit('/')[1],None)
							
							return
						
						xiso_grp.visititems(isotope_dataset_items_visitor)
							
			except Exception as e:
				raise e
			finally:
				if x_file_hdl is not None and x_file_hdl.file != ld_grp.file:
					x_file_hdl.close()
		
		
		for v_iso_dest_path, v_iso_end_offset in v_iso_end_offsets.items():
		
			for v_iso_dset_dest_path, v_iso_dset_source_list in v_iso_dset_sources[v_iso_dest_path].items():
				#print(f'DEBUG : {v_iso_dset_dest_path=}')
				if v_iso_dset_dest_path in ld_grp:
					if not ld_grp[v_iso_dset_dest_path].is_virtual:
						print(f'WARNING : Non-virtual dataset at "{v_iso_dset_dest_path}" in file "{ld_grp.file}", but there are sources that provide data for this dataset. Skipping as we only want to update virtual datasets.')
						continue
					else:
						del ld_grp[v_iso_dset_dest_path]
				
				v_dset_dest_info = v_dset_dest_info_map[v_iso_dset_dest_path]
				
				layout = h5py.VirtualLayout(
					shape=tuple(int(x) for x in v_dset_dest_info.end_offset), 
					dtype=v_dset_dest_info.dtype, 
					maxshape=tuple(None for x in v_dset_dest_info.end_offset)
				)
				
				for v_src_desc in v_iso_dset_source_list:
					#print(f'DEBUG : {v_src_desc=}')
					slice_start = v_src_desc.dest_offset
					slice_end = slice_start + v_src_desc.src_shape
					#print(f'DEBUG : {slice_start=} {slice_end=}')
					shape = tuple(int(s) for s in v_src_desc.src_shape)
					dest_slices = tuple(slice(int(p),int(q)) for p,q in zip(slice_start, slice_end))
					src_slices = tuple(slice(0,s) for s in shape)
					#print(f'DEBUG : {dest_slices=}')
					#print(f'DEBUG : {src_slices=}')
					
					vsource = h5py.VirtualSource(
						v_src_desc.src_file, 
						name=v_src_desc.src_dset_path, 
						shape=shape, 
						dtype=v_dset_dest_info.dtype, 
						maxshape=tuple(None for s in v_src_desc.src_shape)
					)
					layout[*dest_slices] = vsource[*src_slices]
					
				ld_grp.create_virtual_dataset(v_iso_dset_dest_path, layout, fillvalue=v_dset_dest_info.default)
		
		
	


	def consolidate_external_datasets(
		self,
		temp_grp_name : str = 'SCRATCH',
	):
	
		with h5py.File(self.path, 'a') as f:
			s_grp = h5py_helper.ensure_grp(f, 'sources')
			sources = self.get_sources(s_grp)
			
			for s_name, s_file, s_grp_path in sources:
				print(f'DEBUG : {s_name=} {s_file=} {s_grp_path=}')
				if s_file == '.': # these are already in the file, no need to consolidate them
					continue
				
				
				# Copy everything inside the source group to the current file
				if temp_grp_name in s_grp.keys():
					del s_grp[temp_grp_name]
				x_grp = h5py_helper.ensure_grp(s_grp, temp_grp_name)
				
				with h5py.File(Path(f.file.filename).parent / s_file, 'r') as g:
					g.copy(s_grp_path, x_grp, name=x_grp.name + '/line_data')
				
				# delete old source
				del s_grp[s_name]
				
				# move scratch space to replace old source
				s_grp.move(temp_grp_name, s_name)
				
			# loop over all source line data and remove:
			# * any combined isotope groups
			# * any virtual datasets
			# * any empty isotope groups
			# * any empty molecule groups
			# * any empty sources
			for x_name, x_grp in s_grp.items():
				if not isinstance(x_grp, h5py.Group):
					continue
				
				ld_grp = x_grp['line_data']
				for mol_name, mol_grp in ld_grp.items():
					if not isinstance(mol_grp, h5py.Group):
						continue
					
					if self.combined_isotope_group_name in mol_grp:
						del mol_grp[self.combined_isotope_group_name]
					
					for iso_name, iso_grp in mol_grp.items():
						if not isinstance(iso_grp, h5py.Group):
							continue
						
						for dset_name, dset in iso_grp.items():
							if not isinstance(dset, h5py.Dataset):
								continue
							if dset.is_virtual:
								del iso_grp[dset_name]
						
						if len(tuple(iso_grp.keys())) == 0:
							del mol_grp[iso_name]
					if len(tuple(mol_grp.keys())) == 0:
						del ld_grp[mol_name]
				if len(tuple(ld_grp.keys())) == 0:
					del s_grp[x_name]


	def get_sources_grp(self, root_grp : h5py.Group):
		return h5py_helper.ensure_grp(root_grp, 'sources', attrs={'description':'Data for this file split by the source it came from'})
	
	def get_mols_grp(self, x_grp : h5py.Group):
		return h5py_helper.ensure_grp(x_grp, 'molecules', attrs={'description':'Association between RADTRAN molecule ID numbers and molecule names'})
	
	def get_isos_grp(self, x_grp : h5py.Group):
		return h5py_helper.ensure_grp(x_grp, 'isotopologues', attrs={'description':'RADTRAN ID and names of isotopologues'})

	def get_line_data_grp(self, x_grp : h5py.Group):
		return h5py_helper.ensure_grp(x_grp, 'line_data',attrs={'description':'Contains nested tables of line data for each molecule/isotopologue.'})

	def set_source(
			self,
			ldh : LineDataHolder
	):
		with h5py.File(self.path,'a') as f:

			s_grp = self.get_sources_grp(f)

			x_grp = h5py_helper.ensure_grp(s_grp, ldh.name, attrs=dict(description=ldh.description))

			molecules_grp = self.get_mols_grp(x_grp)
			
			molecule_ids = np.array(sorted(list(set([rt_gas_desc.gas_id for rt_gas_desc in ldh.rt_gas_descs]))), dtype=int)
			molecule_names = np.array([RadtranGasDescriptor(id,0).gas_name for id in molecule_ids], dtype='T')
			
			h5py_helper.ensure_dataset(molecules_grp, 'mol_id', data=molecule_ids, maxshape=(None,))
			h5py_helper.ensure_dataset(molecules_grp, 'mol_name', data=molecule_names, maxshape=(None,))
			
			isotopologues_grp = self.get_isos_grp(x_grp)
			
			h5py_helper.ensure_dataset(isotopologues_grp, 'mol_id', data=np.array([rt_gas_desc.gas_id for rt_gas_desc in ldh.rt_gas_descs], dtype=int), maxshape=(None,))
			h5py_helper.ensure_dataset(isotopologues_grp, 'iso_id', data=np.array([rt_gas_desc.iso_id for rt_gas_desc in ldh.rt_gas_descs], dtype=int), maxshape=(None,))
			h5py_helper.ensure_dataset(isotopologues_grp, 'global_iso_id', data=np.array([rt_gas_desc.global_iso_id for rt_gas_desc in ldh.rt_gas_descs], dtype=int), maxshape=(None,))
			h5py_helper.ensure_dataset(isotopologues_grp, 'iso_name', data=np.array([rt_gas_desc.isotope_name for rt_gas_desc in ldh.rt_gas_descs], dtype='T'), maxshape=(None,))
			
			
			d_grp = self.get_line_data_grp(x_grp)
			
			mol_mask = np.ones_like(ldh.mol_id, dtype=bool)
			iso_mask = np.ones_like(ldh.mol_id, dtype=bool)
			
			for rt_gas_desc in ldh.rt_gas_descs:
				mol_mask[...] = ldh.mol_id == rt_gas_desc.gas_id
				iso_mask[...] = mol_mask & (ldh.local_iso_id == rt_gas_desc.iso_id)
				
				mol_grp = h5py_helper.ensure_grp(d_grp, rt_gas_desc.gas_name)
				iso_grp = h5py_helper.ensure_grp(mol_grp, f'{rt_gas_desc.iso_id}')
				
				line_data_table = LineDataTableFormat(
					ldh.mol_id[iso_mask],
					ldh.local_iso_id[iso_mask],
					ldh.nu[iso_mask],
					ldh.sw[iso_mask],
					ldh.a[iso_mask],
					ldh.elower[iso_mask],
					ldh.gamma_self[iso_mask],
					ldh.n_self[iso_mask],
				)
				
				line_data_table.to_hdf5(iso_grp)
				
				b_grp = h5py_helper.ensure_grp(iso_grp, 'broadeners', attrs={'description':'Foreign broadening values for the isotope'})
				if ldh.broadeners is not None:
					for line_broadener_holder in ldh.broadeners:
				
						line_broadener_table = LineBroadenerTableFormat(
							line_broadener_holder.gamma_amb[iso_mask],
							line_broadener_holder.n_amb[iso_mask],
							line_broadener_holder.delta_amb[iso_mask],
						)
						
						amb_grp = h5py_helper.ensure_grp(b_grp, line_broadener_holder.name)
						line_broadener_table.to_hdf5(amb_grp)
					
				
				
		
		self.update_from_sources()

	def update_source(
		self,
		ldh : LineDataHolder
	):
		with h5py.File(self.path,'a') as f:

			s_grp = self.get_sources_grp(f)

			x_grp = h5py_helper.ensure_grp(s_grp, ldh.name, attrs=dict(description=ldh.description))

			molecules_grp = self.get_mols_grp(x_grp)
			
			molecule_ids = np.array(sorted(list(set([rt_gas_desc.gas_id for rt_gas_desc in ldh.rt_gas_descs]))), dtype=int)
			molecule_names = np.array([RadtranGasDescriptor(id,0).gas_name for id in molecule_ids], dtype='T')
			
			mol_id_dset = h5py_helper.get_dataset(molecules_grp, 'mol_id', defaults=dict(shape=(0,), dtype=int, maxshape=(None,)))
			mol_name_dset = h5py_helper.get_dataset(molecules_grp, 'mol_name', defaults=dict(shape=(0,), dtype='T', maxshape=(None,)))
			
			
			mask = np.zeros_like(molecule_ids, dtype=bool)
			for mol_id in mol_id_dset:
				mask |= molecule_ids == mol_id
			mask = ~mask
			
			n_old_mols = mol_id_dset.size
			n_new_mols = np.count_nonzero(mask)
			
			mol_id_dset.resize(n_old_mols + n_new_mols, axis=0)
			mol_name_dset.resize(n_old_mols+n_new_mols, axis=0)
			
			mol_id_dset[n_old_mols:n_old_mols+n_new_mols] = molecule_ids[mask]
			mol_name_dset[n_old_mols:n_old_mols+n_new_mols] = molecule_names[mask]
			
			
			isotopologues_grp = self.get_isos_grp(x_grp)
			
			mol_id_dset = h5py_helper.get_dataset(isotopologues_grp, 'mol_id', defaults=dict(shape=(0,), dtype=int, maxshape=(None,)))
			iso_id_dset = h5py_helper.get_dataset(isotopologues_grp, 'iso_id', defaults=dict(shape=(0,), dtype=int, maxshape=(None,)))
			global_iso_id_dset = h5py_helper.get_dataset(isotopologues_grp, 'global_iso_id', defaults=dict(shape=(0,), dtype=int, maxshape=(None,)))
			iso_name_dset = h5py_helper.get_dataset(isotopologues_grp, 'iso_name', defaults=dict(shape=(0,), dtype='T', maxshape=(None,)))
			
			mol_ids = np.array([rt_gas_desc.gas_id for rt_gas_desc in ldh.rt_gas_descs], dtype=int)
			iso_ids = np.array([rt_gas_desc.iso_id for rt_gas_desc in ldh.rt_gas_descs], dtype=int)
			global_iso_ids = np.array([rt_gas_desc.global_iso_id for rt_gas_desc in ldh.rt_gas_descs], dtype=int)
			iso_names = np.array([rt_gas_desc.isotope_name for rt_gas_desc in ldh.rt_gas_descs], dtype='T')
			
			mask = np.zeros_like(mol_ids, dtype=bool)
			for mol_id in mol_id_dset:
				mask |= mol_ids == mol_id
			mask = ~mask
			
			n_old = mol_id_dset.size
			n_new = np.count_nonzero(mask)
			
			mol_id_dset.resize(n_old+n_new, axis=0)
			mol_id_dset[n_old:n_old+n_new] = mol_ids[mask]
			
			iso_id_dset.resize(n_old+n_new, axis=0)
			iso_id_dset[n_old:n_old+n_new] = iso_ids[mask]
			
			global_iso_id_dset.resize(n_old+n_new, axis=0)
			global_iso_id_dset[n_old:n_old+n_new] = global_iso_ids[mask]
			
			iso_name_dset.resize(n_old+n_new, axis=0)
			iso_name_dset[n_old:n_old+n_new] = iso_names[mask]
			
			
			d_grp = self.get_line_data_grp(x_grp)
			
			mol_mask = np.ones_like(ldh.mol_id, dtype=bool)
			iso_mask = np.ones_like(ldh.mol_id, dtype=bool)
			
			for rt_gas_desc in ldh.rt_gas_descs:
				mol_mask[...] = ldh.mol_id == rt_gas_desc.gas_id
				iso_mask[...] = mol_mask & (ldh.local_iso_id == rt_gas_desc.iso_id)
				
				mol_grp = h5py_helper.ensure_grp(d_grp, rt_gas_desc.gas_name)
				iso_grp = h5py_helper.ensure_grp(mol_grp, f'{rt_gas_desc.iso_id}')
				
				line_data_table = LineDataTableFormat(
					ldh.mol_id[iso_mask],
					ldh.local_iso_id[iso_mask],
					ldh.nu[iso_mask],
					ldh.sw[iso_mask],
					ldh.a[iso_mask],
					ldh.gamma_air[iso_mask],
					ldh.n_air[iso_mask],
					ldh.delta_air[iso_mask],
					ldh.gamma_self[iso_mask],
					ldh.n_self[iso_mask],
					ldh.elower[iso_mask],
					ldh.gp[iso_mask],
					ldh.gpp[iso_mask],
				)
				
				line_data_table.update_hdf5(iso_grp)
		
		self.update_from_sources()


	

	def remove_source(self, name):
		with h5py.File(self.path,'a') as f:
			s_grp = h5py_helper.ensure_grp(f, 'sources')
			
			if name in s_grp:
				del s_grp[name]
		
		self.repack()



	def get_line_data(
			self, 
			mol_name : str, 
			local_iso_id : int,
			ambient_gas : AmbientGas = AmbientGas.AIR,
			iso_keys = (
				'mol_id',
				'local_iso_id',
				'nu',
				'sw',
				'a',
				'elower',
				'gamma_self',
				'n_self',
			),
			broad_keys = (
				'gamma_amb',
				'n_amb',
				'delta_amb',
			),
			wavelength_mask_fn : None | Callable[[np.ndarray], np.ndarray] = None,
			on_missing_mol : Literal['ignore', 'warn', 'error'] = 'error',
			on_missing_iso : Literal['ignore', 'warn', 'error'] = 'warn',
			on_missing_broadener : Literal['ignore', 'warn', 'error'] = 'error',
	) -> tuple[np.ndarray,...]:
	
		null_data = tuple(
			[np.zeros((0,), dtype=LineDataRecordFormat.type(k)) for k in iso_keys] 
			+ [np.zeros((0,), dtype=LineBroadenerRecordFormat.type(k)) for k in broad_keys]
		)
	
		with h5py.File(self.path, 'r') as f:
		
			mask = None
			
			if 'line_data' not in f:
				raise KeyError(f'HDF5 file "{f.file.filename}" does not have a "line_data" group')
			d_grp = f['line_data']
			
			if mol_name not in d_grp:
				match on_missing_mol:
					case 'ignore':
						return null_data
					case 'warn':
						_lgr.warning(f'HDF5 file "{f.file.filename}" does not have a "line_data/{mol_name}" group, returning NULL DATA')
						return null_data
					case _:
						raise KeyError(f'HDF5 file "{f.file.filename}" does not have a "line_data/{mol_name}" group')
			mol_grp = d_grp[mol_name]
			
			if str(local_iso_id) not in mol_grp:
				match on_missing_iso:
					case 'ignore':
						return null_data
					case 'warn':
						_lgr.warning(f'HDF5 file "{f.file.filename}" does not have a "line_data/{mol_name}/{local_iso_id}" group, returning NULL DATA')
						return null_data
					case _:
						raise KeyError(f'HDF5 file "{f.file.filename}" does not have a "line_data/{mol_name}/{local_iso_id}" group')
			iso_grp = mol_grp[str(local_iso_id)]
			
			
			try:
				dsets = [iso_grp[key] for key in iso_keys]
			except Exception as e:
				not_present_keys = tuple(k for k in iso_keys if k not in iso_grp.keys())
				raise KeyError(f'HDF5 file "{f.file.filename}" does not have any of the keys {not_present_keys} in group "line_data/{mol_name}/{local_iso_id}') from e
			
			
			if 'broadeners' not in iso_grp:
				match on_missing_broadener:
					case 'ignore':
						return null_data
					case 'warn':
						_lgr.warning(f'HDF5 file "{f.file.filename}" does not have a "line_data/{mol_name}/{local_iso_id}/broadeners" group, returning NULL DATA')
						return null_data
					case _:
						raise KeyError(f'HDF5 file "{f.file.filename}" does not have a "line_data/{mol_name}/{local_iso_id}/broadeners" group')
			b_grp = iso_grp['broadeners']
			
			if ambient_gas.name not in b_grp:
				match on_missing_broadener:
					case 'ignore':
						return null_data
					case 'warn':
						_lgr.warning(f'HDF5 file "{f.file.filename}" does not have a "line_data/{mol_name}/{local_iso_id}/broadeners/{ambient_gas.name}" group, returning NULL DATA')
						return null_data
					case _:
						raise KeyError(f'HDF5 file "{f.file.filename}" does not have a "line_data/{mol_name}/{local_iso_id}/broadeners/{ambient_gas.name}" group')
			amb_grp = b_grp[ambient_gas.name]
			
			try:
				dsets += [amb_grp[key] for key in broad_keys]
			except Exception as e:
				not_present_keys = tuple(k for k in broad_keys if k not in amb_grp.keys())
				raise KeyError(f'HDF5 file "{f.file.filename}" does not have any of the keys {not_present_keys} in group "line_data/{mol_name}/{local_iso_id}/broadeners/{str(ambient_gas)}') from e
			
			
			if wavelength_mask_fn is not None:
				mask = wavelength_mask_fn(dsets[iso_keys.index('nu')][tuple()]) # this works because we get the isotope datasets first
		
			if mask is None:
				return tuple(dset[tuple()] for dset in dsets)
			else:
				return tuple(dset[mask] for dset in dsets)
			




















