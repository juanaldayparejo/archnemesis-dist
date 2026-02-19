import os
from pathlib import Path
import dataclasses as dc
from typing import Type, Literal#, NamedTuple, Self, Annotated, Callable, Any, Iterable
import textwrap


import h5py


from archnemesis.helpers import h5py_helper
from archnemesis.helpers.h5py_helper import VirtualSourceInfo
from archnemesis.database.datatypes.pf_data.polynomial_pf_data import PolynomialPFData
from archnemesis.database.datatypes.pf_data.tabulated_pf_data import TabulatedPFData
from archnemesis.database.datatypes.pf_list import PFList
from archnemesis.database.data_holders.partition_function_data_holder import PartitionFunctionDataHolder

# Logging
import logging
_lgr = logging.getLogger(__name__)
_lgr.setLevel(logging.INFO)
#_lgr.setLevel(logging.DEBUG)



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
		
		_lgr.info(f'Validation for "{g.name}" in "{g.file.filename}" succeeded')


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
		s_grp : h5py.Group, # /sources group
	) -> list[VirtualSourceInfo,...]:
	
		sources = []
		target_group_name = 'partition_function'
		
		for x_item_name, x_item in s_grp.items():
		
			if isinstance(x_item, h5py.Group): # handle case where '/sources/X' entry is a group, and therefore should have a 'target_group_name' sub-group inside it.
				if target_group_name not in x_item.keys():
					print(f"WARNING : Source group '{x_item.name}' in '{x_item.file.filename}' should have a '{target_group_name}' sub-group. This one has entries {tuple(x_item.keys())}, therefore not using as a source for '/{target_group_name}'.")
					continue
				
				sxpf_item = x_item[target_group_name]
				if isinstance(sxpf_item, h5py.Dataset):
					external_file = '.'
					external_group = target_group_name
					if len(sxpf_item.shape) == 0:
						# dataset is a scalar, so we only have the filename
						external_file = sxpf_item.asstr()[tuple()]
					elif len(sxpf_item.shape) == 1 and sxpf_item.shape[0] == 2: #  dataset is a pair, so we have the filename and the group
						external_file, external_group = (str(x) for x in sxpf_item.astype('T')[:])
					else:
						raise ValueError(f'Dataset "{sxpf_item.name}" in "{sxpf_item.file.filename}" should either be a scalar string, or a string array of shape (2,), but has dtype={x_item.dtype} shape={x_item.shape}')
					
					sources.append(VirtualSourceInfo(sxpf_item.name, external_file, external_group))
				else:
					sources.append(VirtualSourceInfo(x_item.name, '.', x_item.name+f'/{target_group_name}'))
			
			else:
				raise ValueError(f'Expected h5py.Group for entries of "{s_grp.name}" in "{s_grp.file}", but entry "{x_item_name}" has type {type(x_item)}.')
			
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
			
			
			
			