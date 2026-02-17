
import sys
from pathlib import Path
import dataclasses as dc
from typing import NamedTuple, Annotated

import hapi
import archnemesis as ans
import numpy as np
import h5py
from archnemesis.helpers import h5py_helper
from hitran24_isotopes import ISO, hitran_to_radtran, get_isotope_abundances

from archnemesis.enums import AmbientGas
from archnemesis.database.datatypes.gas_descriptor import RadtranGasDescriptor
from archnemesis.database.datatypes.hitran.gas_descriptor import HitranGasDescriptor

from archnemesis.database.offline.ans_line_data_file import AnsLineDataFile, LineDataHolder, LineBroadenerHolder


linedata_file = Path(__file__).parent / 'hitran24.h5'

source_linedata_file = Path(__file__).parent / 'hitran24_test_copy.h5'

LINE_DATA_FROM_HITRAN = False

if LINE_DATA_FROM_HITRAN:
	datadir = str(Path(__file__).parent / 'HITRAN24/')
	table_name = 'hitran24'

	#Initialise the database
	hapi.db_begin(datadir)


	#Reading all the data
	(
		mol_id,
		local_iso_id,
		nu,
		sw,
		a,
		gamma_air,
		gamma_self,
		elower,
		n_air,
		delta_air,
		gp,
		gpp
	) = hapi.getColumns(
		table_name,
		[
			'molec_id',
			'local_iso_id',
			'nu',
			'sw',
			'a',
			'gamma_air',
			'gamma_self',
			'elower',
			'n_air',
			'delta_air',
			'gp',
			'gpp'
		]
	)

	#Identifying unique species
	molec_id_uniq = np.unique(mol_id)
	print(f'{molec_id_uniq=}')

	rt_gas_descs = []
	rt_gas_desc = None
	mask = np.ones_like(mol_id, dtype=bool)
	local_iso_id_radtran = -1*np.ones_like(mol_id, dtype=int)
	mol_id_radtran= -1*np.ones_like(mol_id, dtype=int)

	# Adjust HITRAN data
	for mol_hitran in molec_id_uniq:
		print(f'Adjusting HITRAN data for {mol_hitran=}')
		
		#Getting the isotopic abundances 
		isotopic_abundances = get_isotope_abundances(mol_hitran, ISO)

		#Identifying the isotopes in the database
		iso_id_uniq = np.unique(local_iso_id[mol_id==mol_hitran])

		#Removing the isotopic abundance effect in the line strengths
		
		
		for iso in iso_id_uniq:
			mask[...] = ((mol_id == mol_hitran) & (local_iso_id == iso))
			print(f'{iso=} {np.count_nonzero(mask)=}')
			
			sw[mask] = sw[mask] / isotopic_abundances[iso-1]
			rt_gas_desc = RadtranGasDescriptor(*hitran_to_radtran[(mol_hitran, iso)])
			print(f'{rt_gas_desc=}')
			
			rt_gas_descs.append(rt_gas_desc)
			
			local_iso_id_radtran[mask] = rt_gas_desc.iso_id
			mol_id_radtran[mask] = rt_gas_desc.gas_id


	print(f'{len(rt_gas_descs)=}')
	print(f'{mol_id_radtran.shape=}')
	print(f'{np.count_nonzero(mol_id_radtran<0)=}')

else:
	mol_id_radtran_list = []
	local_iso_id_radtran_list = []
	nu_list = []
	sw_list = []
	a_list = []
	elower_list = []
	gamma_self_list = []
	gamma_air_list = []
	n_air_list = []
	delta_air_list = []

	with h5py.File(source_linedata_file, 'r') as f:
		ld_grp = f['sources/HITRAN24/line_data']
		for mol_grp in ld_grp.values():
			for iso_grp in mol_grp.values():
				mol_id_radtran_list.append(iso_grp['mol_id'][tuple()])
				local_iso_id_radtran_list.append(iso_grp['local_iso_id'][tuple()])
				nu_list.append(iso_grp['nu'][tuple()])
				sw_list.append(iso_grp['sw'][tuple()])
				a_list.append(iso_grp['a'][tuple()])
				elower_list.append(iso_grp['elower'][tuple()])
				gamma_self_list.append(iso_grp['gamma_self'][tuple()])
				
				if 'broadeners' in iso_grp:
					b_grp = iso_grp['broadeners']
					if 'AIR' in b_grp:
						amb_grp = b_grp['AIR']
						gamma_air_list.append(amb_grp['gamma_amb'][tuple()])
						n_air_list.append(amb_grp['n_amb'][tuple()])
						delta_air_list.append(amb_grp['delta_amb'][tuple()])
	
	mol_id_radtran =np.concatenate(mol_id_radtran_list)
	local_iso_id_radtran =np.concatenate(local_iso_id_radtran_list)
	nu =np.concatenate(nu_list)
	sw =np.concatenate(sw_list)
	a =np.concatenate(a_list)
	elower =np.concatenate(elower_list)
	gamma_self =np.concatenate(gamma_self_list)
	gamma_air =np.concatenate(gamma_air_list)
	n_air =np.concatenate(n_air_list)
	delta_air =np.concatenate(delta_air_list)




# Partition function
from archnemesis.database.offline.ans_partition_fn_data_file import AnsPartitionFunctionDataFile, PartitionFunctionDataHolder, TabulatedPFData, PolynomialPFData

mol_list = []
iso_list = []
#temp_list = []
#q_list = []

xx = np.stack((mol_id_radtran, local_iso_id_radtran),axis=1)
print(f'DEBUG : {xx.shape=}')

yy = np.unique(xx, axis=0)
print(f'DEBUG : {yy.shape=}')

pf_data = []

pfdh = PartitionFunctionDataHolder(
	'HITRAN24',
	'Data in this group is taken from the HITRAN24 database',
)

for mol_id_rt, iso_id_rt in yy:
	gas_desc = RadtranGasDescriptor(mol_id_rt, iso_id_rt)
	try:
		ht_gas_desc = HitranGasDescriptor.from_radtran(gas_desc)
	except KeyError:
		continue
	
	try:
		temp = hapi.TIPS_2021_ISOT_HASH[(ht_gas_desc.gas_id,ht_gas_desc.iso_id)]
		q = hapi.TIPS_2021_ISOQ_HASH[(ht_gas_desc.gas_id,ht_gas_desc.iso_id)]
	except KeyError:
		continue
	
	#mol_list.append(np.full_like(temp, fill_value=mol_id_rt, dtype=int))
	#iso_list.append(np.full_like(temp, fill_value=iso_id_rt, dtype=int))
	mol_list.append(mol_id_rt)
	iso_list.append(iso_id_rt)
	
	#temp_list.append(temp)
	#q_list.append(q)
	
	
	if False:
		pfdh.add(
			mol_id_rt,
			iso_id_rt,
			TabulatedPFData(
				temp,
				q
			)
		)
	else:
		pfdh.add(
			mol_id_rt,
			iso_id_rt,
			TabulatedPFData(
				temp,
				q
			).as_poly()
		)

"""
pfdh = PartitionFunctionDataHolder(
	'HITRAN24',
	'Data in this group is taken from the HITRAN24 database',
	#np.array(mol_list, dtype=int),
	#np.array(iso_list, dtype=int),
	
	tuple(pf_data)
)
"""

hitran_pf_data_file = AnsPartitionFunctionDataFile(linedata_file.with_stem('hitran24_pf'))
hitran_pf_data_file.set_source(pfdh)


# Line data
hitran_line_data_holder = LineDataHolder(
	"HITRAN24",
	'Data in this group is taken from the HITRAN24 database',
	mol_id_radtran,
	local_iso_id_radtran,
	nu,
	sw,
	a,
	elower,
	gamma_self,
	np.zeros_like(gamma_self),
	broadeners = [
		LineBroadenerHolder(
			AmbientGas.AIR.name,
			gamma_air,
			n_air,
			delta_air,
		),
	],
)


hitran_line_data_file = AnsLineDataFile(linedata_file)
hitran_line_data_file.set_source(hitran_line_data_holder)




