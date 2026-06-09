
#import sys
import os
from pathlib import Path
#import dataclasses as dc
#from typing import NamedTuple, Annotated

import hapi
#import archnemesis as ans
import numpy as np
import h5py
#from archnemesis.helpers import h5py_helper
from hitran24_isotopes import ISO, hitran_to_radtran, get_isotope_abundances

from archnemesis.enums import AmbientGas
from archnemesis.database.datatypes.gas_descriptor import RadtranGasDescriptor
from archnemesis.database.datatypes.hitran.gas_descriptor import HitranGasDescriptor

from archnemesis.database.data_holders.line_broadener_holder import LineBroadenerHolder
from archnemesis.database.data_holders.line_data_holder import LineDataHolder
from archnemesis.database.data_holders.partition_function_data_holder import PartitionFunctionDataHolder
from archnemesis.database.datatypes.pf_data.tabulated_pf_data import TabulatedPFData
from archnemesis.database.datatypes.pf_data.polynomial_pf_data import PolynomialPFData

from archnemesis.database.filetypes.ans_line_data_file import AnsLineDataFile
from archnemesis.database.filetypes.ans_partition_fn_data_file import AnsPartitionFunctionDataFile


tips_version = "2025"

table_pf_name = 'tips'+tips_version

tabulated_data = False

pf_data = []

pfdh = PartitionFunctionDataHolder(
	'TIPS'+tips_version,
	'Data in this group is taken from the TIPS'+tips_version+' database as implemented in the HAPI library',
)


mol_id, local_iso_id = zip(*ISO.keys())
mol_id = np.array(mol_id,dtype='int32')
local_iso_id = np.array(local_iso_id,dtype='int32')

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

	#Identifying the isotopes in the database
	iso_id_uniq = np.unique(local_iso_id[mol_id==mol_hitran])

	#Removing the isotopic abundance effect in the line strengths
	for iso in iso_id_uniq:
		mask[...] = ((mol_id == mol_hitran) & (local_iso_id == iso))
		print(f'{iso=} {np.count_nonzero(mask)=}')
		
		rt_gas_desc = RadtranGasDescriptor(*hitran_to_radtran[(mol_hitran, iso)])
		print(f'{rt_gas_desc=}')
		
		rt_gas_descs.append(rt_gas_desc)
		
		local_iso_id_radtran[mask] = rt_gas_desc.iso_id
		mol_id_radtran[mask] = rt_gas_desc.gas_id

xx = np.stack((mol_id_radtran, local_iso_id_radtran),axis=1)
yy = np.unique(xx, axis=0)

mol_list = []
iso_list = []

for mol_id_rt, iso_id_rt in yy:
	gas_desc = RadtranGasDescriptor(mol_id_rt, iso_id_rt)
	try:
		ht_gas_desc = HitranGasDescriptor.from_radtran(gas_desc)
	except KeyError:
		continue
	
	try:

		if tips_version == "2021":
			temp = hapi.TIPS_2021_ISOT_HASH[(ht_gas_desc.gas_id,ht_gas_desc.iso_id)]
			q = hapi.TIPS_2021_ISOQ_HASH[(ht_gas_desc.gas_id,ht_gas_desc.iso_id)]
		if tips_version == "2025":
			temp = hapi.TIPS_2025_ISOT_HASH[(ht_gas_desc.gas_id,ht_gas_desc.iso_id)]
			q = hapi.TIPS_2025_ISOQ_HASH[(ht_gas_desc.gas_id,ht_gas_desc.iso_id)]

	except KeyError:
		continue
	
	mol_list.append(mol_id_rt)
	iso_list.append(iso_id_rt)
	
	if tabulated_data:
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
			PolynomialPFData(
				*TabulatedPFData(
					temp,
					q
				).as_poly()
			)
		)

hitran_pf_data_file = AnsPartitionFunctionDataFile(table_pf_name+".h5")
hitran_pf_data_file.add_source_data(pfdh.name, pfdh, pfdh.description)