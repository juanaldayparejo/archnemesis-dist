


import os
from pathlib import Path
from typing import Literal, Any
from contextlib import contextmanager

import h5py

from archnemesis.helpers import h5py_helper
from archnemesis.helpers.h5py_helper import VirtualSourceInfo

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
			# already open in the correct way so just return the file handle
			yield self._file_hdl
			return
			
		elif (self._file_hdl is not None) and (self._file_hdl.mode != mode):
			# Possibly open in incorrect manner, close and re-open if we require extra permissions
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
		with self.open('r') as f:
			print(f'HDF5 File "{f.file.filename}" elements of group "{f.name}"')
			f.visititems(h5py_helper.HDF5Printer())
	
	def _get_sources(
		self,
		s_grp : h5py.Group, # /sources group
	) -> list[VirtualSourceInfo,...]:
	
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
		Look through the "/sources" group and update the molecule/isotope groups in "/partition_function" with virtual
		datasets that reference data defined in "/sources" group.
		"""
		with self.open('a'):

			d_grp = self._get_data_grp(self._file_hdl)

			s_grp = self._get_sources_grp(self._file_hdl)
			
			sources = self._get_sources(s_grp)
			_lgr.debug(f'{sources=}')
					
			self._update_virtual_datasets(d_grp, sources)
			self._validate_data_group(d_grp)
	
	def _get_sources_grp(self, x_grp : h5py.Group):
		if (x_grp.file.mode == 'r'):
			return x_grp['sources']
		else:
			return h5py_helper.ensure_grp(x_grp, 'sources', attrs={'description':'Data for this file split by the source it came from'})


	def _get_data_grp(self, x_grp : h5py.Group):
		if (x_grp.file.mode == 'r'):
			return x_grp[self.target_group_name]
		else:
			return h5py_helper.ensure_grp(x_grp, self.target_group_name, attrs=self.data_grp_attrs)

	def add_source_link(
			self,
			source_name : str,
			fpath : Path, # Path to file
			gpath : None | str = None, # Path to group within source.
	):
		rel_fpath = str(Path(fpath).relative_to(self.path.parent))
		
		with self.open('a'):
			s_grp = self._get_sources_grp(self._file_hdl)
			xs_grp = h5py_helper.ensure_grp(s_grp, source_name)
			
			if gpath is None:
				h5py_helper.ensure_dataset(xs_grp, self.target_group_name, shape=tuple(), data=rel_fpath, dtype='T')
			else:
				h5py_helper.ensure_dataset(xs_grp, self.target_group_name, shape=(2,), data=(rel_fpath, gpath), dtype='T')
		
		self._update_from_sources()
	
	def remove_source(self, name):
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
		if as_virtual:
			ss_grp_name = 'self' # self-source group name
			ss_grp_description = 'Datasets are moved into this group when they are shifted from the data group when using `AnsDatabaseFile.set_data(as_virtual=True,...)`'
			
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
		with self.open('a'):
			s_grp = self._get_sources_grp(self._file_hdl)
			xs_grp = h5py_helper.ensure_grp(s_grp, name, attrs=None if description is None else {'description' : description})
			dxs_grp = self._get_data_grp(xs_grp)
			self._add_data(dxs_grp, data_holder)
		
			self._validate_data_group(dxs_grp)
		
		self._update_from_sources()
	
	
	def _update_virtual_datasets(
			self,
			d_grp : h5py.Group, #"/{target_group_name}" group to update
			sources : list[VirtualSourceInfo,...],
	):
		raise NotImplementedError()
	
	
	def _validate_data_group(self, d_grp : h5py.Group):
		raise NotImplementedError()
	
	
	def _add_data(self, d_grp : h5py.Group, data_holder : Any):
		raise NotImplementedError()
	
	
	def get_data(
			self,
			*args,
			**kwargs
	) -> Any:
		raise NotImplementedError()












