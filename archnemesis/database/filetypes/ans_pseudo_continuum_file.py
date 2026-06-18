from pathlib import Path
#import dataclasses as dc
from typing import Any, Literal#, Type, Literal#, NamedTuple, Self, Annotated, Callable, Any, Iterable


import numpy as np
import h5py

from archnemesis.enum import AmbientGasEnum


from archnemesis.helpers import h5py_helper
from archnemesis.helpers.h5py_helper import VirtualSourceInfo#, VirtualDsetTarget, VirtualGroupTarget

from archnemesis.database.filetypes.ans_base import AnsDatabaseFile
from archnemesis.database.data_holders.pseudo_continuum_data_holder import PseudoContinuumDataHolder

from archnemesis.database.data_layouts.pseudo_continuum_record_layout import PseudoContinuumDataRecordLayout, PseudoContinuumBroadenerRecordLayout
from archnemesis.database.data_layout_writers.pseudo_continuum_table_writer import PseudoContinuumDataTableWriter, PseudoContinuumBroadenerTableWriter

from archnemesis.database.datatypes.pseudo_continuum_data import PseudoContinuumData

# Logging
import logging
_lgr = logging.getLogger(__name__)
#_lgr.setLevel(logging.INFO)
_lgr.setLevel(logging.DEBUG)


P_REF_DEFAULT = 1.0 # atmospheres


class AnsPseudoContinuumFile(AnsDatabaseFile):
	target_group_name = 'pseudo_continuum'
	data_grp_attrs : dict[str, Any] = {
		'description' : 'Contains data that classifies the pseudo-continuum for molecule/isotopologues in a specific temperature range'
	}
	leaf_group_prefix = 'pc_data_'
	leaf_group_idx_fmt = '{:04d}'

	def __init__(
			self,
			path : Path,
	):
		super().__init__(path)
	
	def _update_virtual_datasets(
			self,
			d_grp : h5py.Group, # "/{target_group_name}" group to update
			sources : list[VirtualSourceInfo,...],
	):
		"""
		Update the contents of "/<target_group_name>" from all virtual sources. Only updates virtual datasets, does not update
		non-virtual datasets. Will create virtual datasets for sources that have entries in "/sources/X/<target_group_name>" 
		that do not exist in "/<target_group_name>"
		
		Pseudo-continuum data is made up of wavenumber bins and some values within those bins (strength, broadening coefficients, etc.).
		Therefore, to first order they shouldn't be combined together. Pseudo-continuum groups should be ordered by t_min then wn_min,
		so virtual groups will probably not have the same index as their concrete counterparts.
		"""
		
		source_map = dict()
		
		for s_name, s_file, s_path in sources:
			_lgr.debug(f'{s_name=} {s_file=} {s_path=}')
			x_grp = None
			
			try:
				x_grp = d_grp.file if s_file == '.' else h5py.File(Path(d_grp.file.filename).parent / s_file)
			
				target_grp = x_grp[s_path]
				for mol_grp_name, mol_grp in target_grp.items():
					for iso_grp_name, iso_grp in mol_grp.items():
						print(f'{mol_grp_name=} {iso_grp_name=}')
						
						iso_grp_src_list = []
						for leaf_grp_name, leaf_grp in iso_grp.items():
							print(f'{leaf_grp_name=}')
							if leaf_grp_name.startswith(self.leaf_group_prefix) and isinstance(leaf_grp, h5py.Group):
								iso_grp_src_list.append(
									(
										self._get_pseudo_continuum_parameters(leaf_grp.attrs),
										self._get_virtual_dataset_target_info(
											s_file, 
											leaf_grp,
											item_include_callable = lambda item : (
												(isinstance(item, h5py.Group) and ('/broadeners' in item.name))
												or (
													isinstance(item, h5py.Dataset)
													and (
														(item.name.rsplit('/',1)[1] in PseudoContinuumDataRecordLayout.attrs())
														or (item.name.rsplit('/',1)[1] in PseudoContinuumBroadenerRecordLayout.attrs())
													)
												)
											),
										),
									)
								)
						
						source_map[(mol_grp_name, iso_grp_name)] = iso_grp_src_list
								
			except Exception as e:
				raise e
			finally:
				if s_file != '.':
					x_grp.file.close()
			print('Performing update...')
			for (mol_grp_name, iso_grp_name), iso_grp_src_list in source_map.items():
				print(f'{mol_grp_name=} {iso_grp_name=}')
				mol_grp = h5py_helper.ensure_grp(d_grp, mol_grp_name)
				iso_grp = h5py_helper.ensure_grp(mol_grp, iso_grp_name)
				
				# delete all existing virtual leaf groups
				to_delete_list = []
				for vleaf_grp_name, vleaf_grp in iso_grp.items():
					if (
						vleaf_grp_name.startswith(self.leaf_group_prefix) # test that we are only deleting a group with the correct naming convention
						and vleaf_grp['wn_bin_center'].is_virtual # assume that a single virtual dataset in a group means the whole group is virtual
					):
						to_delete_list.append(vleaf_grp_name)
				
				for to_delete in to_delete_list:
					del iso_grp[to_delete]
				
				print('Deleted existing leaf groups...')
				
				# Order the leaf group sources
				sorted_iso_grp_src_list = sorted(iso_grp_src_list, key= lambda x: x[0])
				print('iso_grp_src_list has been sorted...')
				print(f'{len(sorted_iso_grp_src_list)=}')
				print(f'{len(sorted_iso_grp_src_list[0])=}')
				print(f'{len(sorted_iso_grp_src_list[0][0])=}')
				
				# add sources
				idx = 0
				for leaf_grp_src_pc_parameters, leaf_grp_src_info in sorted_iso_grp_src_list:
					print(f'Adding source {leaf_grp_src_pc_parameters=}')
					vleaf_grp_name = self.get_leaf_grp_name(idx)
					if vleaf_grp_name in iso_grp:
						# if `vleaf_grp_name` is in `iso_grp` at this point, it is because it is a non-virtual group
						# therefore compare it with `leaf_grp_src_order_val` to see if we should be before or after 
						# the concrete group
						if leaf_grp_src_pc_parameters < self._get_pseudo_continuum_parameters(iso_grp[vleaf_grp_name].attrs):
							new_leaf_grp_name = self.get_leaf_grp_name(idx+1)
							iso_grp.move(vleaf_grp_name, new_leaf_grp_name)
						else:
							idx += 1
							vleaf_grp_name = self.get_leaf_grp_name(idx)
					
					print('Write virtual dataset')
					# Write the virtual datsets
					vleaf_grp = h5py_helper.ensure_grp(iso_grp, vleaf_grp_name)
					self._create_virtual_datasets_from(
						vleaf_grp,
						leaf_grp_src_info.vdset_targets,
						leaf_grp_src_info.vsub_grps,
						leaf_grp_src_info.vattrs
					)
	
	
	def _validate_data_group(
			self, 
			d_grp : h5py.Group, # "/{target_group_name}" group to validate
	):
		"""
		Validate the contents of "/<target_group_name>" as much as possible. Should throw an error if any incompatibilities are found.
		"""
		for mol_grp_name, mol_grp in d_grp.items():
			for iso_grp_name, iso_grp in mol_grp.items():
			
				last_leaf_grp_pc_params = (0,0)
				for i, leaf_grp_name, leaf_grp in self.get_increasing_leaf_grp_name_in_grp_iterable(iso_grp):
					
					leaf_grp_pc_params = self._get_pseudo_continuum_parameters(leaf_grp.attrs)
					assert last_leaf_grp_pc_params <= leaf_grp_pc_params, f"Group {leaf_grp.name} was not ordered correctly. Require ({last_leaf_grp_pc_params=}) <= ({leaf_grp_pc_params})"
					
					wn_bin_center_shape = leaf_grp['wn_bin_center'].shape
					for obj_name, obj in leaf_grp.items():
						if isinstance(obj, h5py.Dataset):
							# All datasets should have the same shape
							assert len(obj.shape) == len(wn_bin_center_shape), f"All datasets of {leaf_grp.name} must have the same number of dimensions, but {obj.name} does not."
							assert all(x==y for x,y in zip(obj.shape, wn_bin_center_shape)), f"All datasets of {leaf_grp.name} must have the same shape, but {obj.name} does not."
						elif obj_name == 'broadeners':
							b_grp = obj
							for bx_grp in b_grp.values():
								for b_dset in bx_grp.values():
									if isinstance(b_dset, h5py.Dataset):
										assert len(b_dset.shape) == len(wn_bin_center_shape), f"All datasets of {leaf_grp.name} must have the same number of dimensions, but {b_dset.name} does not."
										assert all(x==y for x,y in zip(b_dset.shape, wn_bin_center_shape)), f"All datasets of {leaf_grp.name} must have the same shape, but {b_dset.name} does not."
					
	def _get_leaf_group_attrs(
			self, 
			data_holder, 
	) -> dict[str,Any]:
		
		return {
			't_cont': data_holder.t_cont, # Temperature at which pseudo-continuum values were computed
			't_unit' : data_holder.t_unit,
			's_max' : data_holder.s_max, # Maximum line strength included in pseudo-continuum
			's_unit' : data_holder.s_unit,
			'p_ref' : data_holder.p_ref,
			'p_unit' : data_holder.p_unit,
			**self.data_grp_attrs
		}
	
	def _get_pseudo_continuum_parameters(
			self, 
			grp_attrs
	)->tuple[float,float]:
		"""
		Leaf groups are ordered first by maximum line strength included in continuum, then by temperature the continuum was calculated at
		
		## RETURNS ##
			grp_pc_parameters : tuple[float,float] - `s_max` and `t_cont` that were used when creating the pseudo-continuum datasets.
		"""
		# TODO: Account for different units
		return (grp_attrs['s_max'], grp_attrs['t_cont'], grp_attrs['p_ref'])
	
	def _select_best_leaf_grp_for_parameters(
			self,
			target_s_max,
			target_temp,
			iso_grp
	) -> tuple[str, h5py.Group , tuple[Any,...]]:
		"""
		A pseudo-continuum is worked out at a specific temperature `t_cont`,
		and made with all the lines that have a strength lower than a maximum value
		`s_max`. When looking up which pseudo-continuum dataset to use we should
		always try and match `s_max`, but if not possible we should use a lower `s_max`,
		but never a higher `s_max` as we don't want to double-count lines. 
		The best `t_cont` to use is the lowest one that is greater than the target temperature.
		
		## RETURNS ##
			leaf_grp_name : str
			leaf_grp : h5py.Group
			leaf_parameters : tuple[Any,...]
		"""
		
		best_grp_name = ''
		best_grp = None
		best_parameters = None
		mismatch_temp = np.inf
		mismatch_s_max = np.inf
		for i, leaf_grp_name, leaf_grp in self.get_increasing_leaf_grp_name_in_grp_iterable(iso_grp):
			s_max, t_ref, p_ref = self._get_pseudo_continuum_parameters(leaf_grp.attrs)
			
			if target_s_max <= 0:
				delta_s_max = 0
			else:
				delta_s_max = target_s_max - s_max
			
			delta_temp = target_temp - t_ref
			
			#print(f'AnsPseudoContinuumFile :: {leaf_grp_name=} {s_max=} {t_ref=} {p_ref=} {delta_s_min=} {delta_temp=} {mismatch_s_max=} {mismatch_temp=}')
			
			if (
				(
					(np.abs(delta_s_max) <= np.abs(mismatch_s_max)) # Want closest `s_max`, must have `s_max` is less than `target_s_max`
					and (
						(delta_s_max >= 0)
					)
				)
				and (
					(np.abs(delta_temp) <= np.abs(mismatch_temp)) # Want closest `temp`, prefer `t_ref` is greater than `target_temp`
					and (
						(delta_temp <= 0)
						or ((delta_temp > 0) and (mismatch_temp > 0))
					)
				)
			):
				mismatch_s_max = delta_s_max
				mismatch_temp = delta_temp
				best_grp_name = leaf_grp_name
				best_grp = leaf_grp
				best_parameters = (s_max, t_ref, p_ref)
		
		return (best_grp_name, best_grp, best_parameters)
	
	
	def _get_target_leaf_group(
			self,
			p_grp : h5py.Group, # "Parent" group that contains leaf groups
			data_holder : PseudoContinuumDataHolder,
	) -> h5py.Group:
		"""
		Each "/pseudo_continuum/<mol>/<iso>" group contains
		a number of "pc_data_XXXX" groups (where XXXX is a numerical ordering)
		the "pc_data_XXXX".
		
		A pseudo-continuum is worked out at a specific temperature `t_cont`,
		and made with all the lines that have a strength lower than a maximum value
		`s_max`. When looking up which pseudo-continuum dataset to use we should
		always try and match `s_max` exactly as otherwise we will either double-count
		or miss out some lines. The best `t_cont` to use is the lowest one that
		is greater than the target temperature.
		
		Therefore, should order by `s_max` then by `t_cont`
		
		When adding data, we must rename other groups to ensure the added
		group has the correct name.
		"""
		
		leaf_grp_attrs = self._get_leaf_group_attrs(data_holder)
		leaf_grp_pc_parameters = self._get_pseudo_continuum_parameters(leaf_grp_attrs)
		
		# work out where this group should go
		leaf_grp_overwrite = False
		leaf_grp_idx = -1
		i=0
		test_pc_grp_name = self.get_leaf_grp_name(i)
		while test_pc_grp_name in p_grp:
			test_grp_attrs = p_grp[test_pc_grp_name].attrs
			test_grp_pc_parameters = self._get_pseudo_continuum_parameters(test_grp_attrs)
			if (leaf_grp_idx < 0) and (leaf_grp_pc_parameters <= test_grp_pc_parameters): # NOTE: `test_grp_pc_parameters` should always be ordered such that the first one `leaf_grp_pc_parameters` is less than is where this data should be inserted.
				# leaf_grp should be inserted before the current test grp so use the current index
				# don't exit, we want to find the largest index
				leaf_grp_idx = i
				"""
				print(f'{(leaf_grp_pc_parameters == test_grp_pc_parameters)=}')
				print(f'{all(k in leaf_grp_attrs for k in test_grp_attrs)=}')
				print(f'{all(k in test_grp_attrs for k in leaf_grp_attrs)=}')
				print(f'{all(leaf_grp_attrs[k] == v for k,v in test_grp_attrs.items())=}')
				print(f'{np.all(p_grp[test_pc_grp_name]['wn_bin_center'].ndim == data_holder.wn_bin_center.ndim)=}')
				print(f'{all(x==y for x,y in zip(p_grp[test_pc_grp_name]['wn_bin_center'].shape, data_holder.wn_bin_center.shape))=}')
				print(f'{np.all(p_grp[test_pc_grp_name]['wn_bin_center'] == data_holder.wn_bin_center)=}')
				print(f'{np.all(p_grp[test_pc_grp_name]['wn_bin_width'] == data_holder.wn_bin_width)=}')
				"""
				
				if ( # Do some tests to see if we should overwrite this group instead
					(leaf_grp_pc_parameters == test_grp_pc_parameters)
					and all(k in leaf_grp_attrs for k in test_grp_attrs)
					and all(k in test_grp_attrs for k in leaf_grp_attrs)
					and all(leaf_grp_attrs[k] == v for k,v in test_grp_attrs.items())
					and np.all(p_grp[test_pc_grp_name]['wn_bin_center'].ndim == data_holder.wn_bin_center.ndim)
					and all(x==y for x,y in zip(p_grp[test_pc_grp_name]['wn_bin_center'].shape, data_holder.wn_bin_center.shape))
					and np.all(p_grp[test_pc_grp_name]['wn_bin_center'] == data_holder.wn_bin_center)
					and np.all(p_grp[test_pc_grp_name]['wn_bin_width'] == data_holder.wn_bin_width)
				):
					_lgr.warn(f'Will overwrite {p_grp[test_pc_grp_name].name} with new data as the attributes and bins are identical between old data and new data.')
					leaf_grp_overwrite = True
			
			i += 1
			test_pc_grp_name = self.get_leaf_grp_name(i)
		
		n_grps = i
		if leaf_grp_idx < 0: # if we are here, the leaf group should be added to the end
			leaf_grp_idx = n_grps
		leaf_grp_name = self.get_leaf_grp_name(leaf_grp_idx)
		
		if leaf_grp_overwrite:
			if leaf_grp_name in p_grp:
				del p_grp[leaf_grp_name]
		else:
			i = n_grps - 1
			
			# shift along all groups with larger indices, work backwards from maximum index (should be `i` at this point)
			while leaf_grp_idx <= i:
				old_pc_grp_name = self.get_leaf_grp_name(i)
				new_pc_grp_name = self.get_leaf_grp_name(i+1)
				p_grp.move(old_pc_grp_name, new_pc_grp_name)
				i-=1
		
		# At this point there should be a "gap" in the leaf groups with the correct name
		# so put our leaf group there.
		
		leaf_grp = h5py_helper.ensure_grp(
			p_grp, 
			leaf_grp_name, 
			attrs = leaf_grp_attrs
		)
		return leaf_grp
	
	def _add_data(
			self, 
			d_grp : h5py.Group, # Usually "/<target_group_name>" or "/sources/X/<target_group_name>"
			data_holder : PseudoContinuumDataHolder,
	):
		"""
		Add data from `data_holder` into `d_grp`
		"""
		mol_mask = np.ones_like(data_holder.mol_id, dtype=bool)
		iso_mask = np.ones_like(data_holder.mol_id, dtype=bool)
		
		for rt_gas_desc in data_holder.rt_gas_descs:
			mol_mask[...] = data_holder.mol_id == rt_gas_desc.gas_id
			iso_mask[...] = mol_mask & (data_holder.local_iso_id == rt_gas_desc.iso_id)
			
			mol_grp = h5py_helper.ensure_grp(d_grp, rt_gas_desc.gas_name)
			iso_grp = h5py_helper.ensure_grp(mol_grp, f'{rt_gas_desc.iso_id}')
			
			leaf_grp = self._get_target_leaf_group(iso_grp, data_holder)
			
			data_table = PseudoContinuumDataTableWriter(
				data_holder.mol_id[iso_mask],
				data_holder.local_iso_id[iso_mask],
				data_holder.wn_bin_center[iso_mask],
				data_holder.wn_bin_width[iso_mask],
				data_holder.line_strength_sum[iso_mask],
				data_holder.line_strength_weighted_mean_lower_energy_state[iso_mask],
				data_holder.line_strength_weighted_gamma_self[iso_mask],
				data_holder.line_strength_weighted_n_self[iso_mask],
			)
			
			data_table.to_hdf5(leaf_grp)
			
			b_grp = h5py_helper.ensure_grp(leaf_grp, 'broadeners', attrs={'description':'Foreign broadening values for the pseudo-continuum'})
			if data_holder.broadeners is not None:
				for line_broadener_part in data_holder.broadeners:
			
					broadener_table = PseudoContinuumBroadenerTableWriter(
						line_broadener_part.line_strength_weighted_gamma_amb[iso_mask],
						line_broadener_part.line_strength_weighted_n_amb[iso_mask],
						#line_broadener_part.line_strength_weighted_delta_amb[iso_mask],
					)
					
					amb_grp = h5py_helper.ensure_grp(b_grp, line_broadener_part.name)
					broadener_table.to_hdf5(amb_grp)
	
	@staticmethod
	def _get_null_data(
			s_max : float,
			t_cont : float,
			p_cont : float,
			requested_wn_range : tuple[float,float],
			n_ambient_gasses : int,
			wn_bin_center : None | np.ndarray = None, # (cm^{-1})
			wn_bin_width : None | float | np.ndarray = 1.0, # (cm^{-1})
	) -> PseudoContinuumData:
		"""
		Return an empty dataset, but it should be ready to accept extra lines if required therefore need to compute a decent value for
		`wn_bin_center` and `wn_bin_width`.
		"""
		#print(f'TESTING: {requested_wn_range=}')
		if wn_bin_width is not None:
			if wn_bin_center is None:
				if isinstance(wn_bin_width, float):
					wn_bin_center = np.arange(*requested_wn_range, wn_bin_width, dtype=float)
				else:
					wn_bin_width_sum = np.cumsum(wn_bin_width)
					wn_bin_center = np.array([requested_wn_range[0] - wn_bin_width[0]/2, *(requested_wn_range[0] - wn_bin_width[0]/2 + wn_bin_width_sum)], dtype=float)
			n_bins = len(wn_bin_center)
		else:
			n_bins = 0
			wn_bin_center = np.zeros((n_bins,), dtype=float)
			wn_bin_width = np.zeros((n_bins,), dtype=float)
		
		return PseudoContinuumData(
			s_max,
			t_cont,
			p_cont,
			requested_wn_range,
			wn_bin_center,
			wn_bin_width,
			np.zeros((n_bins,), dtype=float),
			np.zeros((n_bins,), dtype=float),
			np.zeros((n_bins,), dtype=float),
			np.zeros((n_bins,), dtype=float),
			np.zeros((n_bins,n_ambient_gasses),dtype=float),
			np.zeros((n_bins,n_ambient_gasses),dtype=float),
		)
	
	def _get_broadeners_grp(
			self,
			x_grp, # group to search within, often ".../mol/iso/pc_data_XXXX"
			on_missing_broadener : Literal['ignore', 'warn', 'error'] = 'error',
	) -> h5py.Group:
		if 'broadeners' not in x_grp:
			match on_missing_broadener:
				case 'ignore':
					return None
				case 'warn':
					_lgr.warning(f'HDF5 file "{x_grp.file.filename}" does not have a "{x_grp.name}/broadeners" group, returning NULL DATA')
					return None
				case _:
					raise KeyError(f'HDF5 file "{x_grp.file.filename}" does not have a "{x_grp.name}/broadeners" group')
	
	def _get_single_broadener_grp(
			self,
			x_grp, # group to search within, often ".../mol/iso/pc_data_XXXX/broadeners"
			ambient_gas_name : str,
			on_missing_broadener : Literal['ignore', 'warn', 'error'] = 'error',
	) -> h5py.Group:
		if ambient_gas_name not in x_grp:
			match on_missing_broadener:
				case 'ignore':
					return None
				case 'warn':
					_lgr.warning(f'HDF5 file "{x_grp.file.filename}" does not have a "{x_grp.name}/{ambient_gas_name}" group, returning NULL DATA')
					return None
				case _:
					raise KeyError(f'HDF5 file "{x_grp.file.filename}" does not have a "{x_grp}/{ambient_gas_name}" group')
	
	
	def get_data(
			self,
			mol_name : str, 
			local_iso_id : int,
			temperature : float,
			s_max : float,
			ambient_gasses : AmbientGasEnum | tuple[AmbientGasEnum] = AmbientGasEnum.AIR,
			requested_wn_range : tuple[float,float] = (0, np.inf),
			wn_bin_upper_edge_eta : float = 1E-9,
			on_missing_target : Literal['ignore', 'warn', 'error'] = 'warn',
			on_missing_mol : Literal['ignore', 'warn', 'error'] = 'warn',
			on_missing_iso : Literal['ignore', 'warn', 'error'] = 'warn',
			on_missing_broadener : Literal['ignore', 'warn', 'error'] = 'error',
	) -> PseudoContinuumData:
		"""
		Retrieve stored data from HDF5 file. The exact operations are specific to each type of data.
		
		## ARGUMNETS ##
			mol_name - Name of molecule
			local_iso_id - ID of isotope (RADTRAN ISO ID) in context of parent molecule
			temperature - Temperature to get data for (Kelvin), want to select the pseudo-continuum where `min(t_cont - temperature) and (temperature < t_cont)`. 
			s_max - Maximum line strength included in pesudo-continuum. Should match exactly with line data otherwise will miss or double-count lines.
			ambient_gasses - Tuple of ambient gasses to get broadening data for
			wn_mask_fn - Callable that selects desired wavenumbers (cm^{-1})
			wn_bin_upper_edge_eta -  When testing bin inclusion via `wn_mask_fn`, add this value to the upper edge of a bin to model "less-than" behaviour.
			                         i.e. bins are defined as `lower_edge <= included_wavenumber < upper_edge`. So must add a small amount to the computed
			                         upper edge to ensure some bins are excluded correctly. E.g. if `wn_mask_fn = lambda x: x <= 6`, bins are 2 cm^{-1}
			                         wide and centered on odd numbers. The bin lower edges will be {0,2,4}, the upper edges will be {2,4,6}. Therefore,
			                         when applying `wn_mask_fn` to the upper edges, the last bin will only be correctly excluded if a small number,
			                         `wn_bin_upper_edge_eta` is added to it.
		"""
		#print('AnsPseudoContinuumFile::get_data(...) ARGUMENTS')
		#print(f'\t{mol_name=}')
		#print(f'\t{local_iso_id=}')
		#print(f'\t{temperature=}')
		#print(f'\t{s_max=}')
		#print(f'\t{ambient_gasses=}')
		#print(f'\t{requested_wn_range=}')
		#print(f'\t{wn_bin_upper_edge_eta=}')
		
		if ambient_gasses is None:
			ambient_gasses = tuple()
		elif isinstance(ambient_gasses, AmbientGasEnum):
			ambient_gasses = (ambient_gasses,)
		
		n_ambient_gasses = len(ambient_gasses)
		
		with self.open('r'):
			iso_grp = self._get_data_mol_iso_grp(mol_name, local_iso_id, self._file_hdl, on_missing_target, on_missing_mol, on_missing_iso)
			if iso_grp is None:
				return self._get_null_data(s_max, temperature, P_REF_DEFAULT, requested_wn_range, n_ambient_gasses)
			
			result = None
			
			target_grp_parameters = (s_max, temperature)
			
			leaf_grp_name, leaf_grp, leaf_grp_parameters = self._select_best_leaf_grp_for_parameters(
				*target_grp_parameters,
				iso_grp = iso_grp
			)
			
			if leaf_grp is not None:
				_lgr.info(f'Found compatible data for {s_max=} {temperature=}. Chosen group has {leaf_grp_parameters=}')
				wn_bin_center = leaf_grp['wn_bin_center'][tuple()]
				wn_bin_width = leaf_grp['wn_bin_width'][tuple()]
				wn_bin_lower_edge = wn_bin_center - 0.5*wn_bin_width
				wn_bin_upper_edge = wn_bin_center + 0.5*wn_bin_width + wn_bin_upper_edge_eta
				
				# If any part of a bin is selected by `wn_mask_fn` then return that entire bin
				wn_mask_fn = lambda x: (requested_wn_range[0] <= x) & (x <= requested_wn_range[1])
				wn_mask = wn_mask_fn(wn_bin_lower_edge) | wn_mask_fn(wn_bin_center) | wn_mask_fn(wn_bin_upper_edge)
				n_bins = np.count_nonzero(wn_mask)
				
				if n_bins != 0:
					result = PseudoContinuumData(
						leaf_grp_parameters[0],
						leaf_grp_parameters[1],
						leaf_grp_parameters[2],
						requested_wn_range,
						leaf_grp['wn_bin_center'][wn_mask],
						leaf_grp['wn_bin_width'][wn_mask],
						leaf_grp['line_strength_sum'][wn_mask],
						leaf_grp['line_strength_weighted_mean_lower_energy_state'][wn_mask],
						leaf_grp['line_strength_weighted_gamma_self'][wn_mask],
						leaf_grp['line_strength_weighted_n_self'][wn_mask],
						np.empty((n_bins, n_ambient_gasses), dtype=float),
						np.empty((n_bins, n_ambient_gasses), dtype=float),
					)

					b_grp = self._get_broadeners_grp(leaf_grp, on_missing_broadener)
					if b_grp is None:
						for i, ambient_gas in enumerate(ambient_gasses):
							result.line_strength_weighted_gamma_amb[:,i] = result.line_strength_weighted_gamma_self
							result.line_strength_weighted_n_amb[:,i] = result.line_strength_weighted_n_self
					else:
						for i, ambient_gas in enumerate(ambient_gasses):
							bg_grp = self._get_single_broadener_grp(b_grp, ambient_gas.name, on_missing_broadener)
							if bg_grp is None:
								result.line_strength_weighted_gamma_amb[:,i] = result.line_strength_weighted_gamma_self
								result.line_strength_weighted_n_amb[:,i] = result.line_strength_weighted_n_self
							else:
								result.line_strength_weighted_gamma_amb[:,i] = bg_grp['line_strength_weighted_gamma_amb'][wn_mask]
								result.line_strength_weighted_n_amb[:,i] = bg_grp['line_strength_weighted_n_amb'][wn_mask]
				else:
					_lgr.warn(f'No wavenumbers selected by {wn_mask_fn}, will return empty data. ')
			
			if result is None:
				_lgr.warn(f'No group found that is compatible with {target_grp_parameters=}, will return empty data. ')
				return self._get_null_data(s_max, temperature, P_REF_DEFAULT, requested_wn_range, n_ambient_gasses)
			else:
				return result
			
				
			
		
		
		
		
		
		
		