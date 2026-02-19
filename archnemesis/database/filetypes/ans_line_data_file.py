
import os
from pathlib import Path
from typing import Callable, Literal
import textwrap 

import numpy as np
import h5py

from archnemesis.helpers import h5py_helper
from archnemesis.helpers.h5py_helper import VirtualSourceInfo

from archnemesis.database.datatypes.gas_descriptor import RadtranGasDescriptor
from archnemesis.database.data_holders.line_data_holder import LineDataHolder

from archnemesis.enums import AmbientGas


from archnemesis.database.data_layouts.line_data_record_layout import LineDataRecordLayout
from archnemesis.database.data_layouts.line_broadener_record_layout import LineBroadenerRecordLayout
from archnemesis.database.data_layout_writers.line_broadener_table_writer import LineBroadenerTableWriter
from archnemesis.database.data_layout_writers.line_data_table_writer import LineDataTableWriter

# Logging
import logging
_lgr = logging.getLogger(__name__)
_lgr.setLevel(logging.INFO)
#_lgr.setLevel(logging.DEBUG)




class AnsLineDataFile:
	def __init__(
			self, 
			path : None | Path = None,
	):
		self.path = path
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
					item_type = 'Group' if isinstance(item, h5py.Group) else f'Dataset[{item.shape}, {item.dtype}] = \n{textwrap.indent(str(item[tuple()]), " "*(len(name_tail)+6))}'
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
		
		_lgr.info(f'Validation for "{g.name}" in "{g.file.filename}" succeeded')
		

	def get_sources(
		self,
		s_grp : h5py.Group, # /sources group
	) -> list[VirtualSourceInfo,...]:
	
		sources = []
		target_group_name = 'line_data'
		
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
							
							v_iso_dset_dest_path = f"{xmol_grp_name}/{xiso_grp_name}/{name_tail}"
							v_iso_dset_source_list = v_iso_dset_sources[v_iso_dest_path].setdefault(v_iso_dset_dest_path,[])
							v_shape = np.array(xdset.shape, dtype=int)
							
							v_iso_end_offset = v_iso_end_offsets.setdefault(v_iso_dest_path,np.zeros_like(v_shape, dtype=int))
							if not v_iso_end_updated_once_per_source[v_iso_dest_path]:
								v_iso_end_offset += v_shape
								v_iso_end_updated_once_per_source[v_iso_dest_path] = True
							
							v_iso_dset_source_list.append(
								(
									x_file_name, # src_file
									xdset.name, # src_dset_path
									v_iso_dset_dest_path, # dest_dset_path
									v_shape, # src_shape
									(v_iso_end_offset - v_shape), # dest_offset
								)
							)
							
							v_dset_dest_info_map[v_iso_dset_dest_path] = (v_iso_end_offset, xdset.dtype, self.default_broadening_values.get(xdset.name.rsplit('/')[1],None))
							
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
				
				v_dset_dest_end_offset, v_dset_dest_dtype, v_dset_dest_default = v_dset_dest_info_map[v_iso_dset_dest_path]
				
				layout = h5py.VirtualLayout(
					shape=tuple(int(x) for x in v_dset_dest_end_offset), 
					dtype=v_dset_dest_dtype, 
					maxshape=tuple(None for x in v_dset_dest_end_offset)
				)
				
				for src_file, src_dset_path, dest_dset_path, src_shape, dest_offset in v_iso_dset_source_list:
					#print(f'DEBUG : {v_src_desc=}')
					slice_start = dest_offset
					slice_end = slice_start + src_shape
					#print(f'DEBUG : {slice_start=} {slice_end=}')
					shape = tuple(int(s) for s in src_shape)
					dest_slices = tuple(slice(int(p),int(q)) for p,q in zip(slice_start, slice_end))
					src_slices = tuple(slice(0,s) for s in shape)
					#print(f'DEBUG : {dest_slices=}')
					#print(f'DEBUG : {src_slices=}')
					
					vsource = h5py.VirtualSource(
						src_file, 
						name=src_dset_path, 
						shape=shape, 
						dtype=v_dset_dest_dtype, 
						maxshape=tuple(None for s in src_shape)
					)
					layout[dest_slices] = vsource[src_slices]
					
				ld_grp.create_virtual_dataset(v_iso_dset_dest_path, layout, fillvalue=v_dset_dest_default)


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
				
				line_data_table = LineDataTableWriter(
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
				
						line_broadener_table = LineBroadenerTableWriter(
							line_broadener_holder.gamma_amb[iso_mask],
							line_broadener_holder.n_amb[iso_mask],
							line_broadener_holder.delta_amb[iso_mask],
						)
						
						amb_grp = h5py_helper.ensure_grp(b_grp, line_broadener_holder.name)
						line_broadener_table.to_hdf5(amb_grp)
					
				
				
		
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
			[np.zeros((0,), dtype=LineDataRecordLayout.type(k)) for k in iso_keys] 
			+ [np.zeros((0,), dtype=LineBroadenerRecordLayout.type(k)) for k in broad_keys]
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
			




















