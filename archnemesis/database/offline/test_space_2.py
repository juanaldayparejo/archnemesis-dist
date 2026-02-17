
from pathlib import Path

from typing import Any, NamedTuple, Type
import dataclasses as dc

import os

import numpy as np
import h5py
from archnemesis.helpers import h5py_helper

from ans_line_data_file import AnsLineDataFile
from ans_partition_fn_data_file import AnsPartitionFunctionDataFile

hdf5_file = Path(__file__).parent / "hitran24.h5"


if False:
	# Set isotope "0" to be all of the isotopes combined together
	combined_isotope_group_name = '0'

	with h5py.File(hdf5_file, 'a') as g:

		g = g['linedata']['molecule']
		
		for i, mol_grp_name in enumerate(g.keys()):
			mol_grp = g[mol_grp_name]
			
			layouts = {}
			data_keys = None
			shapes = {}
			
			mol_keys = tuple(k for k in mol_grp.keys() if k != combined_isotope_group_name)
			
			# Get size of data 
			for j, iso_grp_name in enumerate(mol_keys):
				print(f'{mol_grp_name=} {iso_grp_name=}')
				iso_grp = mol_grp[iso_grp_name]
				
				if j==0:
					data_keys = tuple(k for k in iso_grp.keys() if isinstance(iso_grp[k], h5py.Dataset))
				
				for k in data_keys:
					shapes.setdefault(k, [])
					shapes[k].append(iso_grp[k].shape)
			
			if data_keys is None or len(data_keys) <= 0:
				continue
			
			# Create virtual data layout
			for k in data_keys:
				# total shape is sum of all shapes
				assert all(len(shapes[k][0]) == len(s) for s in shapes[k]), "All data keys must have the same number of dimensions"
				assert all(all(q == r for q,r in zip(shapes[k][0],s)) for s in shapes[k]), "All data keys must have the same shape"
				
				total_shape = tuple(np.sum(np.array(shapes[k], dtype=int), axis=0))
				
				layouts[k] = h5py.VirtualLayout(shape=total_shape, dtype=iso_grp[k].dtype, maxshape=(None,))
			
			# Populate layout with data sources
			
			s0 = np.zeros_like(shapes[data_keys[0]][0], dtype=int)
			s1 = np.zeros_like(s0)
			for j, iso_grp_name in enumerate(mol_keys):
				#print(f'Populate layout with data sources{mol_grp_name=} {iso_grp_name=}')
				iso_grp = mol_grp[iso_grp_name]
				
				
				for k in data_keys:
					print(f'{k=} {shapes[k]=}')
					s1[...] = s0 + np.array(shapes[k][j])
					print(f'{s1=}')
					
					slices = tuple(slice(a,h5py.h5s.UNLIMITED) for a,b in zip(s0,s1))
					vsource = h5py.VirtualSource('.', f'/linedata/molecule/{mol_grp_name}/{iso_grp_name}/{k}', shape=shapes[k][j], dtype=iso_grp[k].dtype, maxshape=(None,))
					layouts[k][*slices] = vsource[:h5py.h5s.UNLIMITED]
					
				s0[...] = s1
			
			# Add virtual dataset to molecule
			comb_iso_grp = h5py_helper.ensure_grp(mol_grp, combined_isotope_group_name)
			for name, layout in layouts.items():
				if name in comb_iso_grp:
					del comb_iso_grp[name]
				comb_iso_grp.create_virtual_dataset(name, layout=layout, fillvalue=None)  
					
			

if False:
	with h5py.File(hdf5_file, 'a') as g:
		x = g['/linedata/molecule/H2O/0/nu']
		print(f'{x=}')
		print(f'{x.shape=}')
		if x.is_virtual:
			print(f'{x.virtual_sources()=}')
			
			print(f'{x.virtual_sources()[0][0].shape=}')
			print(f'{x.virtual_sources()[0][3].shape=}')
			print(f'{x.virtual_sources()[0][0].get_select_hyper_blocklist()=}')
			print(f'{x.virtual_sources()[0][3].get_select_hyper_blocklist()=}')


if False:
	
	linedata_file = Path(__file__).parent / 'hitran24.h5'
	with h5py.File(linedata_file, 'a') as f:
		s_grp = h5py_helper.ensure_grp(f, 'sources')
		
		dset = h5py_helper.ensure_dataset(s_grp, 'external_hitran_file', shape=tuple(), data='./hitran24_test_copy.h5', dtype='T')
		
		#if 'external_hitran_group' in s_grp:
		#	del s_grp['external_hitran_group']
		dset = h5py_helper.ensure_dataset(s_grp, 'external_hitran_group', shape=(2,), data=['./hitran24_old_2.h5', '/linedata/molecule'], dtype='T')
	


if True:
	partition_function_data_file_source = Path(__file__).parent / 'hitran24_pf_copy.h5'
	partition_function_data_file_subset = Path(__file__).parent / 'hitran24_pf_external_subset.h5'
	partition_function_data_file_source_subset = Path(__file__).parent / 'hitran24_pf_external_source_subset.h5'
	partition_function_data_file_subset_badname = Path(__file__).parent / 'hitran24_pf_external_subset_badname.h5'
	
	mols_1 = ['CH4', 'H2O']
	
	mols_2 = ['H2S', 'CO', 'CO2']
	
	mols_3 = ['H2', 'GeH4']
	
	mols_4 = ['O2', 'O3']
	
	with h5py.File(partition_function_data_file_source, 'r') as f:
		with h5py.File(partition_function_data_file_subset, 'w') as g:
			xpf_grp = h5py_helper.ensure_grp(g, 'partition_function')
			for mol in mols_1:
				f.copy(f'/sources/HITRAN24/partition_function/{mol}', xpf_grp)
		
		with h5py.File(partition_function_data_file_source_subset, 'w') as g:
			s_grp = h5py_helper.ensure_grp(g, 'sources')
			xs_grp_1 = h5py_helper.ensure_grp(s_grp, 'HITRAN24')
			pfxs_grp_1 = h5py_helper.ensure_grp(xs_grp_1, 'partition_function')
			
			xs_grp_2 = h5py_helper.ensure_grp(s_grp, 'TEST_SOURCE')
			pfxs_grp_2 = h5py_helper.ensure_grp(xs_grp_2, 'partition_function')
			
			for mol in mols_2:
				f.copy(f'/sources/HITRAN24/partition_function/{mol}', pfxs_grp_1)
			
			for mol in mols_3:
				f.copy(f'/sources/HITRAN24/partition_function/{mol}', pfxs_grp_2)
		
		with h5py.File(partition_function_data_file_subset_badname, 'w') as g:
			xpf_grp = h5py_helper.ensure_grp(g, 'part_func_data')
			for mol in mols_4:
				f.copy(f'/sources/HITRAN24/partition_function/{mol}', xpf_grp)
	
	
	partition_function_data_file_external_sources = Path(__file__).parent / 'hitran24_pf_with_external_sources.h5'
	source_map = {
		'external_filename_only' : str(partition_function_data_file_subset.relative_to(partition_function_data_file_external_sources.parent)),
		'external_source_hitran24' : (str(partition_function_data_file_source_subset.relative_to(partition_function_data_file_external_sources.parent)), '/sources/HITRAN24/partition_function'),
		'external_source_test' : (str(partition_function_data_file_source_subset.relative_to(partition_function_data_file_external_sources.parent)), '/sources/TEST_SOURCE/partition_function'),
		'external_file_with_group' : (str(partition_function_data_file_subset_badname.relative_to(partition_function_data_file_external_sources.parent)), '/part_func_data'),
	}
	
	with h5py.File(partition_function_data_file_external_sources, 'w') as f:
		s_grp = h5py_helper.ensure_grp(f, 'sources')
		for source_name, source_info in source_map.items():
			if isinstance(source_info, str):
				h5py_helper.ensure_dataset(s_grp, source_name, shape=tuple(), data=source_info, dtype='T')
			elif isinstance(source_info, tuple) and (len(source_info)==2):
				h5py_helper.ensure_dataset(s_grp, source_name, shape=(2,), data=source_info, dtype='T')
			else:
				raise TypeError(f'{source_name=} {source_info=}. `source_info` should be a string or a tuple of two strings')
	
	pf_file_with_external_sources = AnsPartitionFunctionDataFile(partition_function_data_file_external_sources)
	pf_file_with_external_sources.update_from_sources()
	
	pf_file_with_external_sources.dump()
	


if True:
	line_data_data_file_source = Path(__file__).parent / 'hitran24_copy.h5'
	line_data_data_file_subset = Path(__file__).parent / 'hitran24_external_subset.h5'
	line_data_data_file_source_subset = Path(__file__).parent / 'hitran24_external_source_subset.h5'
	line_data_data_file_subset_badname = Path(__file__).parent / 'hitran24_external_subset_badname.h5'
	
	mols_1 = ['CH4', 'H2O']
	
	mols_2 = ['H2S', 'CO', 'CO2']
	
	mols_3 = ['H2', 'GeH4']
	
	mols_4 = ['O2', 'O3']
	
	with h5py.File(line_data_data_file_source, 'r') as f:
		with h5py.File(line_data_data_file_subset, 'w') as g:
			xpf_grp = h5py_helper.ensure_grp(g, 'line_data')
			for mol in mols_1:
				f.copy(f'/sources/HITRAN24/line_data/{mol}', xpf_grp)
		
		with h5py.File(line_data_data_file_source_subset, 'w') as g:
			s_grp = h5py_helper.ensure_grp(g, 'sources')
			xs_grp_1 = h5py_helper.ensure_grp(s_grp, 'HITRAN24')
			pfxs_grp_1 = h5py_helper.ensure_grp(xs_grp_1, 'line_data')
			
			xs_grp_2 = h5py_helper.ensure_grp(s_grp, 'TEST_SOURCE')
			pfxs_grp_2 = h5py_helper.ensure_grp(xs_grp_2, 'line_data')
			
			for mol in mols_2:
				f.copy(f'/sources/HITRAN24/line_data/{mol}', pfxs_grp_1)
			
			for mol in mols_3:
				f.copy(f'/sources/HITRAN24/line_data/{mol}', pfxs_grp_2)
		
		with h5py.File(line_data_data_file_subset_badname, 'w') as g:
			xpf_grp = h5py_helper.ensure_grp(g, 'lines')
			for mol in mols_4:
				f.copy(f'/sources/HITRAN24/line_data/{mol}', xpf_grp)
	
	
	line_data_data_file_external_sources = Path(__file__).parent / 'hitran24_with_external_sources.h5'
	source_map = {
		'external_filename_only' : str(line_data_data_file_subset.relative_to(line_data_data_file_external_sources.parent)),
		'external_source_hitran24' : (str(line_data_data_file_source_subset.relative_to(line_data_data_file_external_sources.parent)), '/sources/HITRAN24/line_data'),
		'external_source_test' : (str(line_data_data_file_source_subset.relative_to(line_data_data_file_external_sources.parent)), '/sources/TEST_SOURCE/line_data'),
		'external_file_with_group' : (str(line_data_data_file_subset_badname.relative_to(line_data_data_file_external_sources.parent)), '/lines'),
	}
	
	with h5py.File(line_data_data_file_external_sources, 'w') as f:
		s_grp = h5py_helper.ensure_grp(f, 'sources')
		for source_name, source_info in source_map.items():
			if isinstance(source_info, str):
				h5py_helper.ensure_dataset(s_grp, source_name, shape=tuple(), data=source_info, dtype='T')
			elif isinstance(source_info, tuple) and (len(source_info)==2):
				h5py_helper.ensure_dataset(s_grp, source_name, shape=(2,), data=source_info, dtype='T')
			else:
				raise TypeError(f'{source_name=} {source_info=}. `source_info` should be a string or a tuple of two strings')
	
	ld_file_with_external_sources = AnsLineDataFile(line_data_data_file_external_sources)
	ld_file_with_external_sources.update_from_sources()
	
	ld_file_with_external_sources.dump()
	

if False:
	partition_function_data_file = Path(__file__).parent / 'hitran24_pf.h5'
	with h5py.File(linedata_file, 'a') as f:
		s_grp = h5py_helper.ensure_grp(f, 'sources')
		
		dset = h5py_helper.ensure_dataset(s_grp, 'external_hitran_file', shape=tuple(), data='./hitran24_pf_copy.h5', dtype='T')
		
		#if 'external_hitran_group' in s_grp:
		#	del s_grp['external_hitran_group']
		dset = h5py_helper.ensure_dataset(s_grp, 'external_hitran_group', shape=(2,), data=['./hitran24_old_2.h5', '/linedata/molecule'], dtype='T')



if False:

	ld_file = AnsLineDataFile(Path(__file__).parent / 'hitran24.h5')
	
	ld_file.consolidate_external_datasets()
	ld_file.update_from_sources()
	ld_file.repack()



if False:
	import matplotlib.pyplot as plt
	
	with h5py.File(linedata_file, 'r') as f:
		dset = f['/line_data/C2H2/1/nu']
		print(f'{dset.name=} {dset.is_virtual=}')
		
		plt.plot(dset[:])
		plt.show()