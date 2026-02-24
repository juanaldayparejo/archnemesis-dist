
from pathlib import Path
from typing import Callable, Literal, Any

import numpy as np
import h5py

from archnemesis.helpers import h5py_helper
from archnemesis.helpers.h5py_helper import VirtualSourceInfo

from archnemesis.database.data_holders.line_data_holder import LineDataHolder

from archnemesis.enums import AmbientGas

from archnemesis.database.filetypes.ans_base import AnsDatabaseFile
from archnemesis.database.data_layouts.line_data_record_layout import LineDataRecordLayout
from archnemesis.database.data_layouts.line_broadener_record_layout import LineBroadenerRecordLayout
from archnemesis.database.data_layout_writers.line_broadener_table_writer import LineBroadenerTableWriter
from archnemesis.database.data_layout_writers.line_data_table_writer import LineDataTableWriter

# Logging
import logging
_lgr = logging.getLogger(__name__)
_lgr.setLevel(logging.INFO)
#_lgr.setLevel(logging.DEBUG)




class AnsLineDataFile(AnsDatabaseFile):
	target_group_name = 'line_data'
	data_grp_attrs : dict[str, Any] = {
		'description' : "Contains nested tables of line data for each molecule/isotopologue. Some or all entries may be VIRTUAL sources which reference one or more datasets in '/sources/X/line_data'."
	}

	def __init__(
			self, 
			path : None | Path = None,
	):
		super().__init__(path)
		self.default_broadening_values = {
			'gamma_amb' : 1.0,
			'n_amb' : 0.0,
			'delta_amb' : 0.0,
		}


	def _validate_data_group(self, g : h5py.Group):
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
		

	def _update_virtual_datasets(
			self,
			d_grp : h5py.Group,
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

		for src_name, x_file_name, xd_grp_path in sources:
			#print(f'DEBUG : {x_file_name=} {xd_grp_path=} {d_grp.file=} {d_grp.file.filename=}')
			x_file_hdl = None
			
			try:
				x_file_hdl = d_grp if x_file_name == '.' else h5py.File(Path(d_grp.file.filename).parent / x_file_name)
				
				if xd_grp_path not in x_file_hdl:
					raise KeyError(f'No group "{xd_grp_path}" in file "{x_file_hdl.file}".')
				
				xd_grp = x_file_hdl[xd_grp_path]
				
				self._validate_data_group(xd_grp)
				
				# Loop over molecule and isotopologues
				for xmol_grp_name in xd_grp.keys():
					#print(f'DEBUG : AnsLineDataFile.update_virtual_datasets(...) {xmol_grp_name=}')
					xmol_grp = xd_grp[xmol_grp_name]
					
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
				if x_file_hdl is not None and x_file_hdl.file != d_grp.file:
					x_file_hdl.close()
		
		
		for v_iso_dest_path, v_iso_end_offset in v_iso_end_offsets.items():
		
			for v_iso_dset_dest_path, v_iso_dset_source_list in v_iso_dset_sources[v_iso_dest_path].items():
				#print(f'DEBUG : {v_iso_dset_dest_path=}')
				if v_iso_dset_dest_path in d_grp:
					if not d_grp[v_iso_dset_dest_path].is_virtual:
						print(f'WARNING : Non-virtual dataset at "{v_iso_dset_dest_path}" in file "{d_grp.file}", but there are sources that provide data for this dataset. Skipping as we only want to update virtual datasets.')
						continue
					else:
						del d_grp[v_iso_dset_dest_path]
				
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
					
				d_grp.create_virtual_dataset(v_iso_dset_dest_path, layout, fillvalue=v_dset_dest_default)


	def _add_data(
			self,
			d_grp : h5py.Group,
			ldh : LineDataHolder,
	):

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
		return



	def get_data(
			self, 
			mol_name : str, 
			local_iso_id : int,
			ambient_gas : AmbientGas = AmbientGas.AIR,
			*,
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
	
		with self.open('r'):
		
			mask = None
			
			if 'line_data' not in self._file_hdl:
				raise KeyError(f'HDF5 file "{self._file_hdl.file.filename}" does not have a "line_data" group')
			d_grp = self._get_data_grp(self._file_hdl)
			
			if mol_name not in d_grp:
				match on_missing_mol:
					case 'ignore':
						return null_data
					case 'warn':
						_lgr.warning(f'HDF5 file "{self._file_hdl.file.filename}" does not have a "line_data/{mol_name}" group, returning NULL DATA')
						return null_data
					case _:
						raise KeyError(f'HDF5 file "{self._file_hdl.file.filename}" does not have a "line_data/{mol_name}" group')
			mol_grp = d_grp[mol_name]
			
			if str(local_iso_id) not in mol_grp:
				match on_missing_iso:
					case 'ignore':
						return null_data
					case 'warn':
						_lgr.warning(f'HDF5 file "{self._file_hdl.file.filename}" does not have a "line_data/{mol_name}/{local_iso_id}" group, returning NULL DATA')
						return null_data
					case _:
						raise KeyError(f'HDF5 file "{self._file_hdl.file.filename}" does not have a "line_data/{mol_name}/{local_iso_id}" group')
			iso_grp = mol_grp[str(local_iso_id)]
			
			
			try:
				dsets = [iso_grp[key] for key in iso_keys]
			except Exception as e:
				not_present_keys = tuple(k for k in iso_keys if k not in iso_grp.keys())
				raise KeyError(f'HDF5 file "{self._file_hdl.file.filename}" does not have any of the keys {not_present_keys} in group "line_data/{mol_name}/{local_iso_id}') from e
			
			
			if 'broadeners' not in iso_grp:
				match on_missing_broadener:
					case 'ignore':
						return null_data
					case 'warn':
						_lgr.warning(f'HDF5 file "{self._file_hdl.file.filename}" does not have a "line_data/{mol_name}/{local_iso_id}/broadeners" group, returning NULL DATA')
						return null_data
					case _:
						raise KeyError(f'HDF5 file "{self._file_hdl.file.filename}" does not have a "line_data/{mol_name}/{local_iso_id}/broadeners" group')
			b_grp = iso_grp['broadeners']
			
			if ambient_gas.name not in b_grp:
				match on_missing_broadener:
					case 'ignore':
						return null_data
					case 'warn':
						_lgr.warning(f'HDF5 file "{self._file_hdl.file.filename}" does not have a "line_data/{mol_name}/{local_iso_id}/broadeners/{ambient_gas.name}" group, returning NULL DATA')
						return null_data
					case _:
						raise KeyError(f'HDF5 file "{self._file_hdl.file.filename}" does not have a "line_data/{mol_name}/{local_iso_id}/broadeners/{ambient_gas.name}" group')
			amb_grp = b_grp[ambient_gas.name]
			
			try:
				dsets += [amb_grp[key] for key in broad_keys]
			except Exception as e:
				not_present_keys = tuple(k for k in broad_keys if k not in amb_grp.keys())
				raise KeyError(f'HDF5 file "{self._file_hdl.file.filename}" does not have any of the keys {not_present_keys} in group "line_data/{mol_name}/{local_iso_id}/broadeners/{str(ambient_gas)}') from e
			
			
			if wavelength_mask_fn is not None:
				mask = wavelength_mask_fn(dsets[iso_keys.index('nu')][tuple()]) # this works because we get the isotope datasets first
		
			if mask is None:
				return tuple(dset[tuple()] for dset in dsets)
			else:
				return tuple(dset[mask] for dset in dsets)
			




















