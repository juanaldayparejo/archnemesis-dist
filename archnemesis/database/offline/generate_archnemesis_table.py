
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



linedata_file = Path(__file__).parent / 'hitran24.h5'

source_linedata_file = Path(__file__).parent / 'hitran24_test_copy.h5'

LINE_DATA_FROM_HITRAN = True

if LINE_DATA_FROM_HITRAN:
	hitran_path = "/srv/workspace/data/nemesis/spectroscopy/linedata/hitran24/HITRAN24/"
	table_name = 'hitran24'
	table_pf_name = 'tips2025'
	#hitran_path = "/srv/workspace/data/nemesis/spectroscopy/linedata/hitemp/co/hitemp19/"
	#table_name = '05_HITEMP2019'
	datadir = str(Path(hitran_path))



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
		global_upper_quanta,
		global_lower_quanta,
		local_upper_quanta,
		local_lower_quanta,
		ierr,
		iref,
		line_mixing_flag,
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
			'global_upper_quanta',
			'global_lower_quanta',
			'local_upper_quanta',
			'local_lower_quanta',
			'ierr',
			'iref',
			'line_mixing_flag',
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
	guq_list = []
	glq_list = []
	luq_list = []
	llq_list = []
	ierr_list = []
	iref_list = []
	line_mixing_flag_list = []
	gp_list = []
	gpp_list = []

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
	
				guq_list.append(iso_grp['global_upper_quanta'][tuple()])
				glq_list.append(iso_grp['global_lower_quanta'][tuple()])
				luq_list.append(iso_grp['local_upper_quanta'][tuple()])
				llq_list.append(iso_grp['local_lower_quanta'][tuple()])
				ierr_list.append(iso_grp['ierr'][tuple()])
				iref_list.append(iso_grp['iref'][tuple()])
				line_mixing_flag_list.append(iso_grp['line_mixing_flag'][tuple()])
				gp_list.append(iso_grp['gp'][tuple()])
				gpp_list.append(iso_grp['gpp'][tuple()])

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
	global_upper_quanta =np.concatenate(guq_list)
	global_lower_quanta =np.concatenate(glq_list)
	local_upper_quanta =np.concatenate(luq_list)
	local_lower_quanta =np.concatenate(llq_list)
	ierr =np.concatenate(ierr_list)
	iref =np.concatenate(iref_list)
	line_mixing_flag =np.concatenate(line_mixing_flag_list)
	gp =np.concatenate(gp_list)
	gpp =np.concatenate(gpp_list)

global_upper_quanta = np.array(global_upper_quanta, dtype=object)
global_lower_quanta = np.array(global_lower_quanta, dtype=object)
local_upper_quanta  = np.array(local_upper_quanta, dtype=object)
local_lower_quanta  = np.array(local_lower_quanta, dtype=object)
ierr  = np.array(ierr, dtype=object)
iref  = np.array(iref, dtype=object)
line_mixing_flag  = np.array(line_mixing_flag, dtype=object)


# Line data
hitran_line_data_holder = LineDataHolder(
	name = "HITRAN24",
	description = 'Data in this group is taken from the HITRAN24 database',
	mol_id = mol_id_radtran,
	local_iso_id = local_iso_id_radtran,
	nu = nu,
	sw = sw,
	a = a,
	elower = elower,
	gamma_self = gamma_self,
	n_self = np.zeros_like(gamma_self),
	broadeners = [
		LineBroadenerHolder(
			AmbientGas.AIR.name,
			gamma_air,
			n_air,
			delta_air,
		),
	],
	global_upper_quanta = global_upper_quanta,
	global_lower_quanta = global_lower_quanta,
	local_upper_quanta = local_upper_quanta,
	local_lower_quanta = local_lower_quanta,
	ierr = ierr,
	iref = iref,
	line_mixing_flag = line_mixing_flag,
	gp = gp,
	gpp = gpp
)


hitran_line_data_file = AnsLineDataFile(linedata_file)
hitran_line_data_file.add_source_data(hitran_line_data_holder.name, hitran_line_data_holder, hitran_line_data_holder.description)




