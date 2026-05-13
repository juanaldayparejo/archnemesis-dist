


import os
from pathlib import Path
from typing import Literal, Any, Callable
from contextlib import contextmanager

import h5py

from archnemesis.helpers import h5py_helper
from archnemesis.helpers.h5py_helper import VirtualSourceInfo, VirtualDsetTarget, VirtualGroupTarget

import logging
_lgr = logging.getLogger(__name__)
_lgr.setLevel(logging.INFO)
#_lgr.setLevel(logging.DEBUG)


"""
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
			
"""


class AnsDatabaseFile:
	data_grp_attrs : dict[str,Any] = dict()
	target_group_name : str | None = None
	
	def __init__(
			self, 
			path : None | Path = None,
	):
		self.path = path
		
		self._file_hdl : None | h5py.File = None
		
	
	@contextmanager
	def open(
			self,
			mode : Literal['r', 'r+', 'w', 'w-' ,'x' ,'a'] = 'r', # NOTE: this uses [h5py modes](https://docs.h5py.org/en/latest/high/file.html#opening-creating-files) not python ones.
	):
		"""
		Context manager to open HDF5 file. Can open file multiple times and the mode will be the most permissive requested, will only close the file when
		the outermost context is exited.
		"""
	
		if (self._file_hdl is None):
			self._open_mode = mode
			try:
				self._file_hdl = h5py.File(self.path, self._open_mode)
				yield self._file_hdl
			finally:
				self._file_hdl.close()
				self._file_hdl = None
				self._open_mode = None
			return
			
		elif (self._file_hdl is not None) and (self._file_hdl.mode == mode):
			# already open in the correct way so just return the file handle, outer context should handle closing
			yield self._file_hdl
			return
			
		elif (self._file_hdl is not None) and (self._file_hdl.mode != mode):
			# Possibly open in incorrect manner, close and re-open if we require extra permissions
			# outer context should handle closing
			#
			# r  : read
			# r+ : read and write, fail if file does not exist
			# w  : write, truncate file if exists
			# w- : write, fail if file exists (same as 'x')
			# x  : write, fail if file exists (same as 'w-')
			# a  : read and write, create file if it does not exist
			#
			# As this case already has a file handle, the file must exist and be open so the following applies
			# (r) : read permission only
			# (r+, w, w-, x, a) : read-write permission
			#
			# If `mode` is read only, then we can just return the handle without changing anything
			# as the file must already be open with at least read permissions.
			#
			# if `mode` is anything else, we must check that `self._file_hdl.mode` is not 'r', if the
			# file is only open with read permissions, then must re-open with read-write via 'a'.
			if (self._file_hdl.mode == 'r') and (mode != 'r'):
				self._file_hdl.close()
				self._file_hdl = h5py.File(self.path, 'a')
				
			yield self._file_hdl
			return
	
	
	def repack(self):
		"""
		HDF5 does not reduce size when deleting datasets without using `h5repack` utility, however
		`h5py` does not have this capability so we just copy everything to a new file.
		"""
		try:
			temp_file = self.path.with_stem('~'+self.path.stem)
			
			with self.open('r'):
				with h5py.File(temp_file, 'a') as g:
					for v in self._file_hdl.values():
						self._file_hdl.copy(v, g)
			
			os.replace(temp_file, self.path)
		finally:
			if temp_file.exists():
				os.remove(temp_file)
	
	
	def dump(self):
		"""
		Print HDF5 file structure to stdout
		"""
		with self.open('r') as f:
			print(f'HDF5 File "{f.file.filename}" elements of group "{f.name}"')
			f.visititems(h5py_helper.HDF5Printer())
	
	def _get_sources(
		self,
		s_grp : h5py.Group, # usually "/sources" group
	) -> list[VirtualSourceInfo,...]:
		"""
		Get all of the sources in `s_grp` as a list of (src_name, src_file, src_grp) tuples.
		"""
	
		sources = []
		
		for x_item_name, x_item in s_grp.items():
		
			if isinstance(x_item, h5py.Group): # handle case where '/sources/X' entry is a group, and therefore should have a 'self.target_group_name' sub-group inside it.
				if self.target_group_name not in x_item.keys():
					print(f"WARNING : Source group '{x_item.name}' in '{x_item.file.filename}' should have a '{self.target_group_name}' sub-group. This one has entries {tuple(x_item.keys())}, therefore not using as a source for '/{self.target_group_name}'.")
					continue
				
				sxpf_item = x_item[self.target_group_name]
				if isinstance(sxpf_item, h5py.Dataset):
					external_file = '.'
					external_group = self.target_group_name
					if len(sxpf_item.shape) == 0:
						# dataset is a scalar, so we only have the filename
						external_file = sxpf_item.asstr()[tuple()]
					elif len(sxpf_item.shape) == 1 and sxpf_item.shape[0] == 2: #  dataset is a pair, so we have the filename and the group
						external_file, external_group = (str(x) for x in sxpf_item.astype('T')[:])
					else:
						raise ValueError(f'Dataset "{sxpf_item.name}" in "{sxpf_item.file.filename}" should either be a scalar string, or a string array of shape (2,), but has dtype={x_item.dtype} shape={x_item.shape}')
					
					sources.append(VirtualSourceInfo(sxpf_item.name, external_file, external_group))
				else:
					sources.append(VirtualSourceInfo(x_item.name, '.', x_item.name+f'/{self.target_group_name}'))
			
			else:
				raise ValueError(f'Expected h5py.Group for entries of "{s_grp.name}" in "{s_grp.file}", but entry "{x_item_name}" has type {type(x_item)}.')
			
		return sources
	
	
	def _update_from_sources(self):
		"""
		Look through the "/sources" group and update the "/<target_group_name>" group with virtual datasets that
		reference data defined in "/sources/X/<target_group_name>".
		"""
		with self.open('a'):

			d_grp = self._get_data_grp(self._file_hdl)

			s_grp = self._get_sources_grp(self._file_hdl)
			
			sources = self._get_sources(s_grp)
			_lgr.debug(f'{sources=}')
					
			self._update_virtual_datasets(d_grp, sources)
			self._validate_data_group(d_grp)
	
	def _get_sources_grp(
			self, 
			x_grp : h5py.Group, # group to search for ".../sources", is often the root group
	) -> h5py.Group:
		"""
		Retrieve the "/sources" group
		"""
		if (x_grp.file.mode == 'r'):
			return x_grp['sources']
		else:
			return h5py_helper.ensure_grp(x_grp, 'sources', attrs={'description':'Data for this file split by the source it came from'})


	def _get_data_grp(
			self, 
			x_grp : h5py.Group, # group to search for ".../<target_group_name>", is often the root group or a "/sources/<source_name>/" group
	) -> h5py.Group:
		"""
		Retrieve the "data group". E.g. "/line_data" or "/sources/HITRAN/line_data"
		"""
		if (x_grp.file.mode == 'r'):
			return x_grp[self.target_group_name]
		else:
			return h5py_helper.ensure_grp(x_grp, self.target_group_name, attrs=self.data_grp_attrs)


	def _get_data_mol_iso_grp(
			self,
			mol_name : str,
			local_iso_id : int,
			x_grp : h5py.Group, # group to search for ".../<target_group_name>/mol/iso", is often the root group or a "/sources/<source_name>/" group
			on_missing_mol : Literal['ignore', 'warn', 'error'] = 'error',
			on_missing_iso : Literal['ignore', 'warn', 'error'] = 'warn',
	) -> h5py.Group:
		if self.target_group_name not in x_grp:
			raise KeyError(f'HDF5 file "{x_grp.file.filename}" does not have a {self.target_group_name} group')
		d_grp = self._get_data_grp(x_grp)
		
		if mol_name not in d_grp:
			match on_missing_mol:
				case 'ignore':
					return None
				case 'warn':
					_lgr.warning(f'HDF5 file "{x_grp.file.filename}" does not have a "{self.target_group_name}/{mol_name}" group, returning NULL DATA')
					return None
				case _:
					raise KeyError(f'HDF5 file "{x_grp.file.filename}" does not have a "{self.target_group_name}/{mol_name}" group')
		mol_grp = d_grp[mol_name]
		
		if str(local_iso_id) not in mol_grp:
			match on_missing_iso:
				case 'ignore':
					return None
				case 'warn':
					_lgr.warning(f'HDF5 file "{x_grp.file.filename}" does not have a "{self.target_group_name}/{mol_name}/{local_iso_id}" group, returning NULL DATA')
					return None
				case _:
					raise KeyError(f'HDF5 file "{x_grp.file.filename}" does not have a "{self.target_group_name}/{mol_name}/{local_iso_id}" group')
		iso_grp = mol_grp[str(local_iso_id)]
		return iso_grp

	def add_source_link(
			self,
			source_name : str, # Name of the source to create within "/sources"
			fpath : Path, # Path to file
			gpath : None | str = None, # Path to group within source to link to, if `None` will use root of `fpath`
	):
		"""
		Link "/sources/<source_name>" to group `gpath` in file `fpath`
		"""
		rel_fpath = str(Path(fpath).relative_to(self.path.parent))
		
		with self.open('a'):
			s_grp = self._get_sources_grp(self._file_hdl)
			xs_grp = h5py_helper.ensure_grp(s_grp, source_name)
			
			if gpath is None:
				h5py_helper.ensure_dataset(xs_grp, self.target_group_name, shape=tuple(), data=rel_fpath, dtype='T')
			else:
				h5py_helper.ensure_dataset(xs_grp, self.target_group_name, shape=(2,), data=(rel_fpath, gpath), dtype='T')
		
		self._update_from_sources()
	
	def remove_source(
			self, 
			name, # Name of "/source/<name>" to remove
	):
		"""
		Delete source "/source/name", repack file if any removal happened
		"""
		was_source_removed : bool = False
		with self.open('a'):
			s_grp = self._get_sources_grp(self._file_hdl)
			
			if name in s_grp:
				del s_grp[name]
				was_source_removed = True
	
		if was_source_removed:
			self.repack()
	
	def set_data(
			self,
			as_virtual : bool = True,
			data_holder : Any = None
	):
		"""
		Set data from `data_holder` (if not None). 
		If `as_virtual` is True set data from `data_holder` to a virtual group and put the data into "/sources/self". 
		If `as_virtual` is True and there are non-virtual datasets where `data_holder` would go, move the non-virtual datasets into "/sources/self"
		"""
		if as_virtual:
			ss_grp_name = 'self' # self-source group name
			ss_grp_description = 'The source of datasets in this group is the current file. Datasets are moved into this group using `AnsDatabaseFile.set_data(as_virtual=True,...)`, i.e. when a non-virtual "/X/Y" group is re-created as a virtual group, the actual data is moved to "/sources/self/X/Y"'
			
			with self.open('a'):
				d_grp = self._get_data_grp(self._file_hdl)
				s_grp = self._get_sources_grp(self._file_hdl)
				nv_ds_getter = h5py_helper.HDF5GetNonVirtualDatasets()
				
				d_grp.visititems(nv_ds_getter)
				
				if len(nv_ds_getter.non_virtual_dataset_list) > 0:
					ss_grp = h5py_helper.ensure_grp(s_grp, ss_grp_name, attrs={'description' : ss_grp_description})
				
					for dset_name in nv_ds_getter.non_virtual_dataset_list:
						self._file_hdl.move(dset_name, ss_grp.name + dset_name)
				
				if data_holder is not None:
					self.add_source_data(ss_grp_name, data_holder, description=ss_grp_description)
		else:
			self._add_data(self._get_data_grp(self._file_hdl), data_holder)
	
	
	def add_source_data(
			self, 
			name : str, 
			data_holder : Any, 
			description : None | str = None
	):
		"""
		Add data from `data_holder` to the "/sources/<name>" group
		"""
		with self.open('a'):
			s_grp = self._get_sources_grp(self._file_hdl)
			xs_grp = h5py_helper.ensure_grp(s_grp, name, attrs=None if description is None else {'description' : description})
			dxs_grp = self._get_data_grp(xs_grp)
			self._add_data(dxs_grp, data_holder)
		
			self._validate_data_group(dxs_grp)
		
		self._update_from_sources()
	
	def _get_virtual_dataset_target_info(
			self,
			target_file_path_str : str, # Path (as a string) that `x_grp` resides in. Can be relative to the eventual destination HDF5 file, use '.' to denote the destination HDF5 file
			x_grp : h5py.Group, # Group we want to recreate as virtual datasets in the destination file.
			item_include_callable : Callable[[h5py.Group | h5py.Dataset], bool] = lambda x: True, # This callable is passed the HDF5 item path (e.g. "/<group_name_1>/<group_name_2>/<dataset_name>"), if it returns `True` the group or dataset will be included in the result.
	) -> VirtualGroupTarget:
		"""
		Get all the information required to create a virtual group that links to all of the datasets in `x_grp`.
		"""
		x_grp_src_info = VirtualGroupTarget([], dict(), dict(x_grp.attrs))
		for obj_name, obj in x_grp.items():
			#print(f'{obj_name=}')
			#print(f'{item_include_callable(obj)=}')
			if item_include_callable(obj):
				#print('INCLUDED')
				if isinstance(obj, h5py.Dataset):
					#print('DATASET')
					x_grp_src_info.vdset_targets.append(VirtualDsetTarget(target_file_path_str, obj.name, obj.shape, obj.dtype))
				else: # Must be an h5py.Group
					#print('GROUP')
					x_grp_src_info.vsub_grps[obj_name] = self._get_virtual_dataset_target_info(target_file_path_str, obj)
			#else:
			#	print('SKIPPED')
		
		return x_grp_src_info
		
		
	
	def _create_virtual_datasets_from(
			self,
			v_grp, # group that will contain the virtual datasets
			v_dset_info : tuple[VirtualDsetTarget,...],
			v_sub_grp_dict : dict[str, VirtualGroupTarget] = dict(),
			v_grp_attrs : dict[str,Any] = dict(),
	):
		#print(f'{v_grp=}')
		#print(f'{v_dset_info=}')
		#print(f'{v_sub_grp_dict=}')
		#print(f'{v_grp_attrs=}')
		
		for k,v in v_grp_attrs.items():
			v_grp.attrs[k] = v
		
		for vdt in v_dset_info:
			layout = h5py.VirtualLayout(
				shape = vdt.shape,
				dtype = vdt.dtype,
			)
			vsource = h5py.VirtualSource(
				vdt.file,
				name = vdt.path,
				shape = vdt.shape,
				dtype = vdt.dtype,
			)
			layout[...] = vsource[...]
			
			v_grp.create_virtual_dataset(vdt.path.rsplit('/',1)[1], layout, fillvalue=None)
		
		for v_sub_grp_name, (v_sub_grp_dset_info, v_sub_sub_grp_dict, v_sub_grp_attrs) in v_sub_grp_dict.items():
			self._create_virtual_datasets_from(
				h5py_helper.ensure_grp(v_grp, v_sub_grp_name),
				v_sub_grp_dset_info,
				v_sub_sub_grp_dict,
				v_sub_grp_attrs,
			)
	
	def _update_virtual_datasets(
			self,
			d_grp : h5py.Group, # "/{target_group_name}" group to update
			sources : list[VirtualSourceInfo,...],
	):
		"""
		Update the contents of "/<target_group_name>" from all virtual sources. Only updates virtual datasets, does not update
		non-virtual datasets. Will create virtual datasets for sources that have entries that do not exist in "/<target_group_name>"
		"""
		raise NotImplementedError()
	
	
	def _validate_data_group(
			self, 
			d_grp : h5py.Group, # "/{target_group_name}" group to validate
	):
		"""
		Validate the contents of "/<target_group_name>" as much as possible. Should throw an error if any incompatibilities are found.
		"""
		raise NotImplementedError()
	
	
	def _add_data(
			self, 
			d_grp : h5py.Group, # Usually "/<target_group_name>" or "/sources/X/<target_group_name>"
			data_holder : Any,
	):
		"""
		Add data from `data_holder` into `d_grp`
		"""
		raise NotImplementedError()
	
	
	def get_data(
			self,
			*args,
			**kwargs
	) -> Any:
		"""
		Retrieve stored data from HDF5 file. The exact operations are specific to each type of data.
		"""
		raise NotImplementedError()












