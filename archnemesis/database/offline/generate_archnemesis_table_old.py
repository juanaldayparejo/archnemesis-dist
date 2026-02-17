
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

from archnemesis.database.datatypes.gas_descriptor import RadtranGasDescriptor
from archnemesis.database.datatypes.hitran.gas_descriptor import HitranGasDescriptor



from structured_qualifiers import IDNumber, Quantity
from record_format import RecordFormat
from table_format import TableFormat


LINE_DATA_SOURCES = [
	'UNKNOWN',
	'HITRAN24',
	'HITEMP',
	'EXOMOL',
]







class LineDataRecordFormat(RecordFormat):
	mol_id       : Annotated[int,   IDNumber('RADTRAN ID of molecule')]
	local_iso_id : Annotated[int,   IDNumber('Isotope ID in context of parent molecule')]
	nu           : Annotated[float, Quantity('Wavenumber of line', 'cm^{-1}')]
	sw           : Annotated[float, Quantity('Line intensity at T = 296 K', 'cm^{-1}/(molec.cm^{-2})')]
	a            : Annotated[float, Quantity('Einstein A-coefficient', 's^{-1}')]
	gamma_air    : Annotated[float, Quantity('Air-broadened Lorentzian half-width at half-maximum at p = 1 atm and T = 296 K', 'cm{^-1} atm^{-1}')]
	gamma_self   : Annotated[float, Quantity('Self-broadened HWHM at 1 atm pressure and 296 K', 'cm{^-1} atm^{-1}')]
	elower       : Annotated[float, Quantity('Lower-state energy', 'cm^{-1}')]
	n_air        : Annotated[float, Quantity('Temperature exponent for the air-broadened HWHM', 'NUMBER')]
	delta_air    : Annotated[float, Quantity('Pressure shift induced by air, referred to p=1 atm', 'cm^{-1} atm^{-1}')]
	gp           : Annotated[float, Quantity('Upper state degeneracy', 'NUMBER')]
	gpp          : Annotated[float, Quantity('Lower state degeneracy', 'NUMBER')]
	source       : Annotated[int,   IDNumber('The source id of values for this line data entry')]





print(f'{LineDataRecordFormat}')


class LineDataTableFormat(TableFormat):
	record_format : type[RecordFormat] = LineDataRecordFormat
	
	def to_hdf5(self, grp : h5py.Group):
		for i, attr in enumerate(self.__slots__):
			#print(f'LineDataTableFormat.to_hdf5(...) {grp=} {i=} {attr=} {len(getattr(self, attr))=}')
			dset = h5py_helper.ensure_dataset(grp, attr, data=getattr(self, attr), dtype=self.record_format.type(attr), maxshape=(None,))
			for k, v in self.record_format.metadata(attr).as_dict().items():
				dset.attrs[k] = v



datadir = str(Path(__file__).parent / 'HITRAN24/')
table_name = 'hitran24'
linedata_file = Path(__file__).parent / 'hitran24.h5'

source_id = LINE_DATA_SOURCES.index('HITRAN24')


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





#Writing HDF5 file
with h5py.File(linedata_file,'a') as f:

	m_grp = h5py_helper.ensure_grp(f, 'meta')
	
	if '/sources' in m_grp:
		del m_grp['/sources']
	
	sources = np.array(LINE_DATA_SOURCES, dtype='T')
	dset = h5py_helper.ensure_dataset(m_grp, 'sources', data=sources, dtype=sources.dtype, maxshape=(None,))
	dset.attrs['description'] = 'Names of sources of line data, each line has a source it came from'
		
	
	
	
	ld_grp = h5py_helper.ensure_grp(f, 'linedata')
	
	d_grp = h5py_helper.ensure_grp(ld_grp, 'molecule')
	
	mol_mask = np.ones_like(mol_id_radtran, dtype=bool)
	iso_mask = np.ones_like(mol_id_radtran, dtype=bool)
	
	for rt_gas_desc in rt_gas_descs:
		mol_mask[...] = mol_id_radtran == rt_gas_desc.gas_id
		iso_mask[...] = mol_mask & (local_iso_id_radtran == rt_gas_desc.iso_id)
		
		mol_grp = h5py_helper.ensure_grp(d_grp, rt_gas_desc.gas_name)
		iso_grp = h5py_helper.ensure_grp(mol_grp, f'{rt_gas_desc.iso_id}')
		
		line_data_table = LineDataTableFormat(
			mol_id_radtran[iso_mask],
			local_iso_id_radtran[iso_mask],
			nu[iso_mask],
			sw[iso_mask],
			a[iso_mask],
			gamma_air[iso_mask],
			gamma_self[iso_mask],
			elower[iso_mask],
			n_air[iso_mask],
			delta_air[iso_mask],
			gp[iso_mask],
			gpp[iso_mask],
			np.ones((np.count_nonzero(iso_mask),), dtype=int) * source_id,
		)
		
		line_data_table.to_hdf5(iso_grp)

# Delete variables we no longer need
del mol_id
del local_iso_id
del nu
del sw
del a
del gamma_air
del gamma_self
del elower
del n_air
del delta_air
del gp
del gpp


# Set isotope "0" to be all of the isotopes combined together
combined_isotope_group_name = '0'

with h5py.File(linedata_file, 'a') as g:

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