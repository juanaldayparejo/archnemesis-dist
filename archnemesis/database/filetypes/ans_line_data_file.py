
from pathlib import Path
from typing import Literal, Any

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

from archnemesis.database.datatypes.line_set_data import LineSetData

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
	leaf_group_prefix = 'line_set_'
	leaf_group_idx_fmt = '{:04d}'

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
				
				for i, leaf_grp_name, leaf_grp in self.get_increasing_leaf_grp_name_in_grp_iterable(iso_grp):
					leaf_grp.visititems(EachChildDatasetHasSameShapeVisitor())
		
		_lgr.info(f'Validation for "{g.name}" in "{g.file.filename}" succeeded')
		

	def _update_virtual_datasets(
			self,
			d_grp : h5py.Group,
			sources : list[VirtualSourceInfo,...],
	):
		"""
		Update the contents of "/line_data" from all virtual sources. Only updates virtual datasets, does not update
		non-virtual datasets. Will create virtual datasets for sources that have entries that do not exist in "/line_data".
		
		TODO: Have a think on if we want to combine data from all `s_min==0` and `t_ref_1 == t_ref_2` groups for a specific
		      isotope together.
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
										self._get_line_set_parameters(leaf_grp.attrs),
										self._get_virtual_dataset_target_info(
											s_file, 
											leaf_grp,
											item_include_callable = lambda item : (
												(isinstance(item, h5py.Group) and ('/broadeners' in item.name))
												or (
													isinstance(item, h5py.Dataset)
													and (
														(item.name.rsplit('/',1)[1] in LineDataRecordLayout.attrs())
														or (item.name.rsplit('/',1)[1] in LineBroadenerRecordLayout.attrs())
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
			#print('Performing update...')
			for (mol_grp_name, iso_grp_name), iso_grp_src_list in source_map.items():
				#print(f'{mol_grp_name=} {iso_grp_name=}')
				mol_grp = h5py_helper.ensure_grp(d_grp, mol_grp_name)
				iso_grp = h5py_helper.ensure_grp(mol_grp, iso_grp_name)
				
				# delete all existing virtual leaf groups
				to_delete_list = []
				for vleaf_grp_name, vleaf_grp in iso_grp.items():
					if (
						vleaf_grp_name.startswith(self.leaf_group_prefix) # test that we are only deleting a group with the correct naming convention
						and vleaf_grp['nu'].is_virtual # assume that a single virtual dataset in a group means the whole group is virtual
					):
						to_delete_list.append(vleaf_grp_name)
				
				for to_delete in to_delete_list:
					del iso_grp[to_delete]
				
				#print('Deleted existing leaf groups...')
				
				# Order the leaf group sources
				sorted_iso_grp_src_list = sorted(iso_grp_src_list, key= lambda x: x[0])
				#print('iso_grp_src_list has been sorted...')
				#print(f'{len(sorted_iso_grp_src_list)=}')
				#print(f'{len(sorted_iso_grp_src_list[0])=}')
				#print(f'{len(sorted_iso_grp_src_list[0][0])=}')
				
				# add sources
				idx = 0
				for leaf_grp_src_parameters, leaf_grp_src_info in sorted_iso_grp_src_list:
					#print(f'Adding source {leaf_grp_src_parameters=}')
					vleaf_grp_name = self.get_leaf_grp_name(idx)
					if vleaf_grp_name in iso_grp:
						# if `vleaf_grp_name` is in `iso_grp` at this point, it is because it is a non-virtual group
						# therefore compare it with `leaf_grp_src_order_val` to see if we should be before or after 
						# the concrete group
						if leaf_grp_src_parameters < self._get_line_set_parameters(iso_grp[vleaf_grp_name].attrs):
							new_leaf_grp_name = self.get_leaf_grp_name(idx+1)
							iso_grp.move(vleaf_grp_name, new_leaf_grp_name)
						else:
							idx += 1
							vleaf_grp_name = self.get_leaf_grp_name(idx)
					
					#print('Write virtual dataset')
					# Write the virtual datsets
					vleaf_grp = h5py_helper.ensure_grp(iso_grp, vleaf_grp_name)
					self._create_virtual_datasets_from(
						vleaf_grp,
						leaf_grp_src_info.vdset_targets,
						leaf_grp_src_info.vsub_grps,
						leaf_grp_src_info.vattrs
					)

	def _get_leaf_group_attrs(
			self, 
			data_holder, 
	) -> dict[str,Any]:
		
		return {
			't_ref': data_holder.t_ref, # Temperature at which pseudo-continuum values were computed
			't_unit' : data_holder.t_unit,
			's_min' : data_holder.s_min, # Maximum line strength included in pseudo-continuum
			's_unit' : data_holder.s_unit,
			'p_ref' : data_holder.p_ref,
			'p_unit' : data_holder.p_unit,
			**self.data_grp_attrs
		}

	def _get_line_set_parameters(
			self, 
			grp_attrs
	)->tuple[float,float,float]:
		"""
		Leaf groups are ordered first by minimum line strength included in in the set, then by reference temperature that the set data was calculated at
		
		## RETURNS ##
			grp_line_set_parameters : tuple[float,float] - `s_min` and `t_ref` that were used when creating this line set
		"""
		return (grp_attrs['s_min'], grp_attrs['t_ref'], grp_attrs['p_ref'])
	
	def _select_best_line_set_for_parameters(
			self,
			target_s_min,
			target_temp,
			iso_grp
	) -> tuple[str, h5py.Group , tuple[Any,...]]:
		"""
		A line set is worked out at a specific temperature `t_ref`,
		and made with all the lines that have a strength higher than a minimum value
		`s_min`. 
		
		When looking up which line set dataset to use we should
		always try and match `s_min` exactly as otherwise we will either double-count
		or miss out some lines. 
		
		The best `t_ref` to use is the lowest one that
		is greater than the target temperature, but if no temperature greater is available
		fallback to closest temperature. 
		
		Special Cases:
		`s_min == 0` - we can use any temperature as there is no minimum strength
		`s_min < 0` - We don't care about minimum strength so find best matching temperature
		
		## RETURNS ##
			best_grp_name : str
			best_grp : h5py.Group
			best_parameters : tuple[Any,...]
		"""
		best_grp_name = ''
		best_grp = None
		best_parameters = None
		mismatch_temp = np.inf
		mismatch_s_min = np.inf
		for i, leaf_grp_name, leaf_grp in self.get_increasing_leaf_grp_name_in_grp_iterable(iso_grp):
			s_min, t_ref, p_ref = self._get_line_set_parameters(leaf_grp.attrs)
			
			if target_s_min <= 0:
				delta_s_min = 0
			else:
				delta_s_min = target_s_min - s_min
			
			delta_temp = target_temp - t_ref
			
			#print(f'AnsLineDataFile :: {leaf_grp_name=} {s_min=} {t_ref=} {p_ref=} {delta_s_min=} {delta_temp=} {mismatch_s_min=} {mismatch_temp=}')
			
			if (
				(delta_s_min <= mismatch_s_min)
				and (
					(np.abs(delta_temp) <= np.abs(mismatch_temp))
					and (
						(delta_temp <= 0)
						or ((delta_temp > 0) and (mismatch_temp > 0))
					)
				)
			):
				mismatch_s_min = delta_s_min
				mismatch_temp = delta_temp
				best_grp_name = leaf_grp_name
				best_grp = leaf_grp
				best_parameters = (s_min, t_ref, p_ref)
		
		return (best_grp_name, best_grp, best_parameters)
				
			

	def _get_target_leaf_group(
			self,
			p_grp : h5py.Group, # "Parent" group that contains leaf groups
			data_holder : LineDataHolder,
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
		leaf_grp_parameters = self._get_line_set_parameters(leaf_grp_attrs)
		
		# work out where this group should go
		leaf_grp_overwrite = False
		leaf_grp_idx = -1
		i=0
		test_grp_name = self.get_leaf_grp_name(i)
		while test_grp_name in p_grp:
			test_grp_attrs = p_grp[test_grp_name].attrs
			test_grp_parameters = self._get_line_set_parameters(test_grp_attrs)
			if (leaf_grp_idx < 0) and (leaf_grp_parameters <= test_grp_parameters): # NOTE: `test_grp_pc_parameters` should always be ordered such that the first one `leaf_grp_pc_parameters` is less than is where this data should be inserted.
				# leaf_grp should be inserted before the current test grp so use the current index
				# don't exit, we want to find the largest index
				leaf_grp_idx = i
				
				if ( # Do some tests to see if we should overwrite this group instead
					(leaf_grp_parameters == test_grp_parameters)
				):
					_lgr.warn(f'Will overwrite {p_grp[test_grp_name].name} with new data as the attributes and bins are identical between old data and new data.')
					leaf_grp_overwrite = True
			
			i += 1
			test_grp_name = self.get_leaf_grp_name(i)
		
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
			d_grp : h5py.Group, # Usually "/line_data" or "/sources/X/line_data"
			data_holder : LineDataHolder,
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
			
			line_data_table = LineDataTableWriter(
				data_holder.mol_id[iso_mask],
				data_holder.local_iso_id[iso_mask],
				data_holder.nu[iso_mask],
				data_holder.sw[iso_mask],
				data_holder.a[iso_mask],
				data_holder.elower[iso_mask],
				data_holder.gamma_self[iso_mask],
				data_holder.n_self[iso_mask],
			)
			
			line_data_table.to_hdf5(leaf_grp)
			
			b_grp = h5py_helper.ensure_grp(leaf_grp, 'broadeners', attrs={'description':'Foreign broadening values for the isotope'})
			if data_holder.broadeners is not None:
				for line_broadener_holder in data_holder.broadeners:
			
					line_broadener_table = LineBroadenerTableWriter(
						line_broadener_holder.gamma_amb[iso_mask],
						line_broadener_holder.n_amb[iso_mask],
						line_broadener_holder.delta_amb[iso_mask],
					)
					
					amb_grp = h5py_helper.ensure_grp(b_grp, line_broadener_holder.name)
					line_broadener_table.to_hdf5(amb_grp)
		return

	def _get_null_data(
			self,
			s_min : float,
			t_ref : float,
			p_ref : float,
			requested_wn_range : tuple[float,float],
			n_broadeners : int
	):
		line_fields_to_populate = tuple(x for x in LineSetData._fields if x in LineDataRecordLayout.attrs())
		broadener_felds_to_populate = tuple(x for x in LineSetData._fields if x in LineBroadenerRecordLayout.attrs())
		return LineSetData(
			s_min,
			t_ref,
			p_ref,
			requested_wn_range,
			*(np.empty((0,), dtype=LineDataRecordLayout.type(x)) for x in line_fields_to_populate),
			*(np.empty((0,n_broadeners), dtype=LineBroadenerRecordLayout.type(x)) for x in broadener_felds_to_populate),
		)
	
	def _get_broadeners_grp(
			self,
			x_grp, # group to search within, often ".../mol/iso/leaf_group_XXXX"
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
			x_grp, # group to search within, often ".../mol/iso/leaf_group_XXXX/broadeners"
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
			s_min : float = -1,
			temperature : float = 0,
			ambient_gasses : AmbientGas | tuple[AmbientGas] = (AmbientGas.AIR,),
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
			requested_wn_range : tuple[float,float] = (0, np.inf),
			on_missing_target : Literal['ignore', 'warn', 'error'] = 'warn',
			on_missing_mol : Literal['ignore', 'warn', 'error'] = 'warn',
			on_missing_iso : Literal['ignore', 'warn', 'error'] = 'warn',
			on_missing_broadener : Literal['ignore', 'warn', 'error'] = 'error',
	) -> LineSetData:
		print('AnsLineDataFile::get_data(...)')
	
		if ambient_gasses is None:
			ambient_gasses = tuple()
		elif isinstance(ambient_gasses, AmbientGas):
			ambient_gasses = (ambient_gasses,)
		
		n_ambient_gasses = len(ambient_gasses)
		
		line_fields_to_populate = tuple(x for x in LineSetData._fields if x in LineDataRecordLayout.attrs())
		broadener_felds_to_populate = tuple(x for x in LineSetData._fields if x in LineBroadenerRecordLayout.attrs())
		
		with self.open('r'):
			iso_grp = self._get_data_mol_iso_grp(mol_name, local_iso_id, self._file_hdl, on_missing_target, on_missing_mol, on_missing_iso)
			if iso_grp is None:
				return self._get_null_data(s_min,temperature,1,requested_wn_range,n_ambient_gasses)
			
			result = None
		
			mask = None
			
			target_line_set_params = (s_min, temperature)
			print(f'AnsLineDataFile :: {target_line_set_params=}')
			
			leaf_grp_name, leaf_grp, leaf_grp_parameters = self._select_best_line_set_for_parameters(
				*target_line_set_params,
				iso_grp = iso_grp
			)
			
			if leaf_grp is not None:
				mask = (requested_wn_range[0] <= leaf_grp['nu'][tuple()]) & (leaf_grp['nu'][tuple()] <= requested_wn_range[1])
				n_lines = np.count_nonzero(mask)
				#print(f'{mol_name=} {local_iso_id=} {n_lines=}')
				if n_lines > 0:
					#print(f'{leaf_grp_parameters=}')
					#print(f'{line_fields_to_populate=}')
					#print(f'{broadener_felds_to_populate=}')
					result = LineSetData(
						leaf_grp_parameters[0],
						leaf_grp_parameters[1],
						leaf_grp_parameters[2],
						requested_wn_range,
						*(leaf_grp[x][mask] for x in line_fields_to_populate),
						*(np.empty((n_lines, n_ambient_gasses), dtype=LineBroadenerRecordLayout.type(x)) for x in broadener_felds_to_populate)
					)
					
					b_grp = self._get_broadeners_grp(leaf_grp, on_missing_broadener)
					if b_grp is None:
						for name in broadener_felds_to_populate:
							getattr(result, name).fill(self.default_broadening_values[name])
					else:
						for i, ambient_gas in enumerate(ambient_gasses):
							bg_grp = self._get_single_broadener_grp(b_grp, ambient_gas.name, on_missing_broadener)
							if bg_grp is None:
								for name in broadener_felds_to_populate:
									getattr(result, name)[:,i].fill(self.default_broadening_values[name])
							else:
								for name in broadener_felds_to_populate:
									getattr(result, name)[:,i] = bg_grp[name][mask]
				else:
					_lgr.warn(f'Compatible group "{leaf_grp.name}" found, but no lines selected by {requested_wn_range=}. Therefore will return empty data.')
			
			if result is None:
				_lgr.warn(f'No compatible group found for {target_line_set_params=}. Therefore will return empty data.')
				return self._get_null_data(s_min, temperature, 1, requested_wn_range, n_ambient_gasses)
			else:
				return result
			




















