import os
from pathlib import Path
import dataclasses as dc
from typing import NamedTuple, Type, Self, Literal#, Annotated, Callable, Any, Iterable
import textwrap


import numpy as np
import h5py


from archnemesis.helpers import h5py_helper
from archnemesis.database.datatypes.gas_descriptor import RadtranGasDescriptor

# Logging
import logging
_lgr = logging.getLogger(__name__)
#_lgr.setLevel(logging.INFO)
_lgr.setLevel(logging.DEBUG)


@dc.dataclass(slots=True)
class PFData:
	
	def __post_init__(self):
		self.init_domain()
	
	def init_domain(self):
		raise NotImplementedError
	
	def in_domain(self, T : float | np.ndarray) -> bool | np.ndarray:
		return ((self.domain[0] <= T) & (T <= self.domain[1]))
	
	def as_structured_array(self) -> np.ndarray:
		raise NotImplementedError
	
	def __call__(self, T: float | np.ndarray) -> float | np.ndarray:
		raise NotImplementedError
	
	def as_table(self, t_array : None | np.ndarray = None) -> tuple[np.ndarray, np.ndarray]:
		raise NotImplementedError

@dc.dataclass(slots=True)
class TabulatedPFData(PFData):
	temp : np.ndarray
	q : np.ndarray
	domain : np.ndarray = dc.field(default_factory=lambda : np.array([0,np.inf],dtype=float))
	
	
	def init_domain(self):
		min, max = (np.min(self.temp), np.max(self.temp))
		if self.domain[0] < min:
			self.domain[0] = min
		if self.domain[1] > max:
			self.domain[1] = max
		return
	
	def as_structured_array(self) -> np.ndarray:
		return np.array(
			[
				self.temp, 
				self.q
			], 
			dtype=[('temp',float), ('q',float)]
		)
	
	def __call__(self, T : float | np.ndarray) -> float | np.ndarray:
		return np.interp(T, self.temp, self.q)
	
	def as_table(self, t_array : None | np.ndarray = None) -> Self:
		if t_array is None:
			return TabulatedPFData(self.temp, self.q, self.domain) 
		else:
			return TabulatedPFData(t_array, np.interp(t_array, self.temp, self.q), np.array([np.min(t_array), np.max(t_array)], dtype=float))
		
	
	def as_poly(self, t_array : None | np.ndarray = None, n : int = 5) -> PFData:
		instance = self.as_table(t_array)
		return PolynomialPFData(np.polynomial.Polynomial.fit(instance.temp, instance.q, deg=n).coef, instance.domain)

@dc.dataclass(slots=True)
class PolynomialPFData(PFData):
	coeffs : np.ndarray
	domain : np.ndarray = dc.field(default_factory=lambda : np.array([0,np.inf],dtype=float))
	
	_poly : np.polynomial.Polynomial = dc.field(init=False, repr=False)
	
	def init_domain(self):
		return
	
	
	
	def __setattr__(self, name, value):
		if name == 'coeffs':
			#self.coeffs = value
			super(PolynomialPFData, self).__setattr__(name, value) # have to add arguments to `super()` due to using `slots=True`
			self._poly = np.polynomial.Polynomial(self.coeffs, symbol='T')
		else:
			super(PolynomialPFData, self).__setattr__(name, value)

	def as_structured_array(self) -> np.ndarray:
		return np.array(
			[
				self._poly.coef
			], 
			dtype=[('coeffs',float, self.poly.coef.shape)]
		)
	
	def __call__(self, T : float | np.ndarray) -> float | np.ndarray:
		return self._poly(T)
	
	def as_table(self, t_array : np.ndarray = None) -> PFData:
		if t_array is None:
			t_array = np.linspace(50, 500, 1000)
		return TabulatedPFData(t_array, self._poly(t_array), np.array([np.min(t_array), np.max(t_array)], dtype=float))
	
	def as_poly(self, t_array : None | np.ndarray = None, n : None | int = None) -> Self:
		if n is None:
			return PolynomialPFData(self.coeffs, self.domain)
		
		return PolynomialPFData(np.polynomial.Polynomial.fit(*self.as_table(t_array), deg=n).coef, self.domain)


@dc.dataclass(slots=True)
class PFList:
	pf_data_list : list[PFData,...] = dc.field(default_factory=list)
	
	def append(self, pf_data : PFData) -> Self:
		self.pf_data_list.append(pf_data)
		return self
	
	def __call__(self, T : float | np.ndarray) -> float | np.ndarray:
		if len(self.pf_data_list) == 0:
			raise RuntimeError(f'{self} has no partition data.')
	
		if isinstance(T, np.ndarray):
			acc = np.zeros_like(T, float)
			n = np.zeros_like(T, int)
			for pf_data in self.pf_data_list:
				mask = pf_data.in_domain(T)
				acc[mask] += pf_data(T[mask])
				n[mask] += 1
			
			if np.any(~np.nonzero(n)):
				raise RuntimeError(f'Partition functions do not completely cover the range of temperatures provided. Missing data for T={T[~np.nonzero(n)]}')
		
		else:
			acc = 0
			n = 0
			for pf_data in self.pf_data_list:
				#print(f'DEBUG : {pf_data=} {acc=} {n=} {T=}')
				if pf_data.in_domain(T):
					acc += pf_data(T)
					n += 1
			
			if n == 0:
				raise RuntimeError(f'Partition functions do not cover the requested temperature {T=}.')
		
		return acc/n
		

@dc.dataclass(slots=True)
class PartitionFunctionDataHolder:
	# source information
	name : str # Name of source, will result in a "/sources/X" group
	description : str # Description of source, will be the "description" attribute of the "/sources/X" group
	
	# A dictionary of partition function data. Keys are `RadtranGasDescriptor` instances, 
	# values are a list of `PFData` instances. `PFData` sub-classes define a callable class that
	# returns the partition function of an isotopologue at a specified temperature for a specified
	# temperature domain.
	# Will define the "/sources/X/partition_function/<mol_name>/<iso_id>/pf_data_0000"
	# groups that hold data for each of the `PFData` instances in the list.
	data : dict[RadtranGasDescriptor, list[PFData,...]] = dc.field(default_factory=dict) 
	
	def add(self, mol_id, local_iso_id, pf_data : PFData):
		self.data.setdefault(RadtranGasDescriptor(mol_id, local_iso_id), []).append(pf_data)
	
	def items(self):
		yield from self.data.items()



class VirtualSourceInfo(NamedTuple):
	src_name : str
	src_file : str
	src_grp : str


class AnsPartitionFunctionDataFile:
	pf_data_types : tuple[Type,...] = (TabulatedPFData, PolynomialPFData)

	def __init__(
			self,
			path : Path,
	):
		self.path = path

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

	def validate_partition_function_group(self, g : h5py.Group):
		
		class EachImmediateChildGroupIsLikeAPFDataInstanceVisitor:
			def __init__(self):
				pass
			
			def __call__(self, name_tail : str, item : h5py.Group | h5py.Dataset):
				if (name_tail.count('/') == 0) and name_tail.startswith('pf_data_') and isinstance(item, h5py.Group):
					# We should be a `pf_data_0000` group
					try:
						int(name_tail[len('pf_data_'):])
					except Exception as e:
						raise ValueError(f'Group "{item.name}" in file "{item.file.filename}" must end with an integer denoting the index of the partition function data group.') from e
					
					_lgr.debug(f'Validated that {item.name} is a partition function data group')
					
					# Therefore, the group should be some type of `PFData` instance
					
					try:
						pf_type = h5py_helper.get_dataset(
							item, 
							'partition_function_type', 
							defaults=dict(shape=tuple(), dtype='T', attrs={'description' : "Describes how the partition function is specified"}),
							on_is_not_dataset = 'error',
							on_missing = 'error',
						).asstr()[tuple()]
					except (TypeError, KeyError) as e:
						raise ValueError(f'Partition function data group "{item.name}" in HDF5 file "{item.file.filename}" must have a "partition_function_type" dataset.') from e
					
					pf_data_type_names = [typ.__name__ for typ in AnsPartitionFunctionDataFile.pf_data_types]
					
					if pf_type in pf_data_type_names:
						typ = AnsPartitionFunctionDataFile.pf_data_types[pf_data_type_names.index(pf_type)]
						for field in dc.fields(typ):
							if field.name.startswith('_'):
								continue
							
							try:
								h5py_helper.get_dataset(
									item, 
									field.name, 
									on_is_not_dataset='error', 
									on_missing='error'
								)
							except Exception as e:
								missing_keys = tuple(field.name for field in dc.fields(typ) if field.name not in item)
								raise KeyError(f'Group "{item.name}" in file "{item.file.filename}" has dataset "{item.name}/partition_function_type" = "{pf_type}". Therefore must have entries {tuple(dc.fields(typ))} but are missing entries {missing_keys}') from e
					else:
						raise ValueError(f'Dataset "{item.name}/partition_function_type" in file "{item.file.filename}" must be one of {pf_data_type_names} not "{pf_type}"')
					
					_lgr.debug(f'Validated that {item.name} contains items that describe an instance of a sub-class of PFData')
				
				else: # we are not a `pf_data_0000` group
					return
				
		
		for mol_grp_name, mol_grp in g.items():
			if isinstance(mol_grp, h5py.Dataset):
				raise TypeError(f'Item "{mol_grp.name}" in "{g.file.filename}" is a dataset. Expected that all direct children of group "{g.name}" are groups not datasets.')
			
			for iso_grp_name, iso_grp in mol_grp.items():
				if isinstance(iso_grp, h5py.Dataset):
					raise TypeError(f'Item "{iso_grp.name}" in "{g.file.filename}" is a dataset. Expected that all direct children of "molecule_group" group "{mol_grp.name}" are groups not datasets.')
				
				iso_grp.visititems(EachImmediateChildGroupIsLikeAPFDataInstanceVisitor())
		
		_lgr.debug(f'Validation for "{g.name}" in "{g.file.filename}" succeeded')


	def update_from_sources(self):
		"""
		Look through the "/sources" group and update the molecule/isotope groups in "/partition_function" with virtual
		datasets that reference data defined in "/sources" group.
		"""
		with h5py.File(self.path, 'a') as f:

			pf_grp = self.get_pf_grp(f)

			s_grp = self.get_sources_grp(f)
			
			sources = self.get_sources(s_grp)
			_lgr.debug(f'{sources=}')
					
			self.update_virtual_datasets(pf_grp, sources)
			self.validate_partition_function_group(pf_grp)

	def get_sources(
		self,
		s_grp : h5py.Group,
	) -> list[VirtualSourceInfo,...]:
	
		sources = []
		
		for x_item_name, x_item in s_grp.items():
		
			if isinstance(x_item, h5py.Dataset):
				external_file = '.'
				external_group = 'partition_function'
				if len(x_item.shape) == 0:
					 # dataset is a scalar, so we only have the filename
					 external_file = x_item.asstr()[tuple()]
				elif len(x_item.shape) == 1 and x_item.shape[0] == 2: #  dataset is a pair, so we have the filename and the group
					external_file, external_group = (str(x) for x in x_item.astype('T')[:])
				else:
					raise ValueError(f'Dataset "{x_item.name}" in "{x_item.file.filename}" should either be a scalar string, or a string array of shape (2,), but has dtype={x_item.dtype} shape={x_item.shape}')
				
				sources.append(VirtualSourceInfo(x_item.name, external_file, external_group))
			
			elif isinstance(x_item, h5py.Group): # handle case where '/sources/X' entry is a group, and therefore should have a 'partition_function' sub-group inside it.
				if 'partition_function' not in x_item.keys():
					print(f"WARNING : Source group '{x_item.name}' in '{x_item.file.filename}' should have a 'partition_function' sub-group. This one has entries {tuple(x_item.keys())}, therefore not using as a source.")
					continue
				sources.append(VirtualSourceInfo(x_item.name, '.', x_item.name+'/partition_function'))
			
			else:
				raise ValueError(f'Expected h5py.Group or h5py.Dataset for entries of {s_grp.name} in {s_grp.file}, but entry {x_item_name} has type {type(x_item)}.')
			
		return sources


	def update_virtual_datasets(
			self,
			pf_grp : h5py.Group, #"/partition_function" group to update
			sources : list[VirtualSourceInfo,...],
	):
		"""
		Update the contents of "/partition_function" from all virtual sources. Only updates virtual datasets, does not update
		non-virtual datasets. Will create virtual datasets for sources that have entries that do not exist in "/partition_function"
		
		Partition function entries are just sequentially numbered "pf_data_" groups inside each ".../mol_name/iso_id" path, therefore can just
		create each group in turn and populate as a virtual dataset, will need to re-number the "pf_data_" groups, but that is not a problem.
		"""
		
		iso_pf_source_map = dict()
		
		for s_name, s_file, s_pf_path in sources:
			_lgr.debug(f'{s_name=} {s_file=} {s_pf_path=}')
			x_grp = None
			
			try:
				x_grp = pf_grp.file if s_file == '.' else h5py.File(Path(pf_grp.file.filename).parent / s_file)
			
				xpf_grp = x_grp[s_pf_path]
				for mol_grp_name, mol_grp in xpf_grp.items():
					for iso_grp_name, iso_grp in mol_grp.items():
						for pf_data_grp_name, pf_data_grp in iso_grp.items():
							if not pf_data_grp_name.startswith('pf_data_') or not isinstance(pf_data_grp, h5py.Group):
								continue # is not a `pf_data_grp`
							
							
							pf_data_dsets = []
							for pf_dset_name, pf_dset in pf_data_grp.items():
								if not isinstance(pf_dset, h5py.Dataset):
									continue # is not a pf_dset
								
								pf_data_dsets.append(
									(s_file, pf_dset.name, pf_dset.shape, pf_dset.dtype)
								)
							
							
							iso_pf_source_map.setdefault((mol_grp_name, iso_grp_name), []).append(tuple(pf_data_dsets))
			
			
			
			except Exception as e:
				raise e
			finally:
				if s_file != '.':
					x_grp.file.close()
			
			for (mol_grp_name, iso_grp_name), pf_data_src_grp_list in iso_pf_source_map.items():
				_lgr.debug(f'{mol_grp_name=} {iso_grp_name=}')
				mol_grp = h5py_helper.ensure_grp(pf_grp, mol_grp_name)
				iso_grp = h5py_helper.ensure_grp(mol_grp, iso_grp_name)
				
				# delete all existing virtual pf_data_ groups
				to_delete_list = []
				for pf_data_grp_name, pf_data_grp in iso_grp.items():
					if 'partition_function_type' in pf_data_grp:
						if pf_data_grp['partition_function_type'].is_virtual:
							to_delete_list.append(pf_data_grp_name)
				
				for to_delete in to_delete_list:
					del iso_grp[to_delete]
				
				# add sources
				idx = 0
				for pf_data_dsets in pf_data_src_grp_list:
					
					while f'pf_data_{idx:04d}' in iso_grp:
						idx += 1
					pf_data_grp_name = f'pf_data_{idx:04d}'
					pf_data_grp = h5py_helper.ensure_grp(iso_grp, pf_data_grp_name)
					
					for s_file, pf_dset_path, pf_dset_shape, pf_dset_dtype in pf_data_dsets:
						#_lgr.debug(f'{s_file=} {pf_dset_path=} {pf_dset_shape=} {pf_dset_dtype=}')
						layout = h5py.VirtualLayout(
							shape = pf_dset_shape,
							dtype = pf_dset_dtype,
						)
						vsource = h5py.VirtualSource(
							s_file,
							name = pf_dset_path,
							shape = pf_dset_shape,
							dtype=pf_dset_dtype,
						)
						layout[...] = vsource[...]
						
						pf_data_grp.create_virtual_dataset(pf_dset_path.rsplit('/',1)[1], layout, fillvalue=None)
	
	
	def get_sources_grp(self, root_grp : h5py.Group):
		return h5py_helper.ensure_grp(root_grp, 'sources', attrs={'description':'Data for this file split by the source it came from'})
	
	def get_mols_grp(self, x_grp : h5py.Group):
		return h5py_helper.ensure_grp(x_grp, 'molecules', attrs={'description':'Association between RADTRAN molecule ID numbers and molecule names'})
	
	def get_isos_grp(self, x_grp : h5py.Group):
		return h5py_helper.ensure_grp(x_grp, 'isotopologues', attrs={'description':'RADTRAN ID and names of isotopologues'})
	
	def get_pf_grp(self, x_grp : h5py.Group):
		return h5py_helper.ensure_grp(x_grp, 'partition_function',attrs={'description':'Contains data that classifies the partition function for molecule/isotopologues'})
		
	def set_source(
			self,
			pfdh : PartitionFunctionDataHolder
	):
		"""
		Set sub-group in "/sources" group with data specified in `pfdh`
		"""
		with h5py.File(self.path, 'a') as f:
			s_grp = self.get_sources_grp(f)
			x_grp = h5py_helper.ensure_grp(s_grp, pfdh.name, attrs=dict(description=pfdh.description))
			pf_grp = self.get_pf_grp(x_grp)
			
			for rt_gas_desc, pf_data_list in pfdh.items():
				
				mol_grp = h5py_helper.ensure_grp(pf_grp, rt_gas_desc.gas_name)
				iso_grp = h5py_helper.ensure_grp(mol_grp, f'{rt_gas_desc.iso_id}')
				
				for pf_data_idx, pf_data in enumerate(pf_data_list):
					pf_data_grp = h5py_helper.ensure_grp(iso_grp, f'pf_data_{pf_data_idx:04d}', attrs={'description' : 'An instance of partition function data for this isotopologue'})
					
					pf_type = pf_data.__class__.__name__
					if isinstance(pf_data, self.pf_data_types):
						for field in dc.fields(pf_data):
							if field.name.startswith('_'):
								continue
							_lgr.info(f'{field=}')
							h5py_helper.ensure_dataset(pf_data_grp, field.name, data=getattr(pf_data, field.name))
					else:
						raise TypeError(f'`pfdh.data` must be an instance of one of the following types {tuple(typ for typ in self.pf_data_types)}, not {type(pf_data)}.')
				
					h5py_helper.ensure_dataset(pf_data_grp, 'partition_function_type', shape=tuple(), data=pf_type, dtype='T', attrs={'description' : "Describes how the partition function is specified"})
		
		self.update_from_sources()
	
	
	
	def get_partition_function_data(
			self,
			mol_name : str, 
			local_iso_id : int,
			on_missing_mol : Literal['ignore', 'warn', 'error'] = 'error',
			on_missing_iso : Literal['ignore', 'warn', 'error'] = 'warn',
	) -> PFList:
		#print(f'DEBUG : AnsPartitionFunctionDataFile.get_partition_function_data(...) {mol_name=} {local_iso_id=} {on_missing_mol=} {on_missing_iso=}')
		null_data = None
		
		with h5py.File(self.path, 'r') as f:
			
			if 'partition_function' not in f:
				raise KeyError(f'HDF5 file "{f.file.filename}" does not have a "partition_function" group')
			pf_grp = self.get_pf_grp(f)
			
			if mol_name not in pf_grp:
				match on_missing_mol:
					case 'ignore':
						return null_data
					case 'warn':
						_lgr.warning(f'HDF5 file "{f.file.filename}" does not have a "partition_function/{mol_name}" group, returning NULL DATA')
						return null_data
					case _:
						raise KeyError(f'HDF5 file "{f.file.filename}" does not have a "partition_function/{mol_name}" group')
			mol_grp = pf_grp[mol_name]
			
			if str(local_iso_id) not in mol_grp:
				match on_missing_iso:
					case 'ignore':
						return null_data
					case 'warn':
						_lgr.warning(f'HDF5 file "{f.file.filename}" does not have a "partition_function/{mol_name}/{local_iso_id}" group, returning NULL DATA')
						return null_data
					case _:
						raise KeyError(f'HDF5 file "{f.file.filename}" does not have a "partition_function/{mol_name}/{local_iso_id}" group')
			iso_grp = mol_grp[str(local_iso_id)]
			
			pf_list = PFList()
			
			for pf_data_grp_name, pf_data_grp in iso_grp.items():
				if not pf_data_grp_name.startswith('pf_data_') or not isinstance(pf_data_grp, h5py.Group):
					continue # is not a `pf_data_` group so skip it.
				
				try:
					pf_type = h5py_helper.get_dataset(
						pf_data_grp, 
						'partition_function_type', 
						defaults=dict(shape=tuple(), dtype='T', attrs={'description' : "Describes how the partition function is specified"}),
						on_is_not_dataset = 'error',
						on_missing = 'error',
					).asstr()[tuple()]
				except (TypeError, KeyError) as e:
					raise ValueError(f'Partition function data group "{pf_data_grp.name}" in HDF5 file "{pf_data_grp.file.filename}" must have a "partition_function_type" dataset.') from e
				
				pf_data_type_names = [typ.__name__ for typ in self.pf_data_types]
				
				if pf_type in pf_data_type_names:
					typ = self.pf_data_types[pf_data_type_names.index(pf_type)]
					pf_data = typ(
						**dict(
							(
								field.name,
								h5py_helper.get_dataset(
									pf_data_grp, 
									field.name, 
									on_is_not_dataset='error', 
									on_missing='error'
								)[tuple()]
							) for field in dc.fields(typ) if not field.name.startswith('_')
						)
					)
				else:
					raise ValueError(f'`pf_type`="{pf_type}" must be one of {pf_data_type_names}.')
				
				#print(f'DEBUG : {pf_data=}')
				pf_list.append(pf_data)
				
			return pf_list
			
			
			
			