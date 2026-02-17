

from typing import Annotated, TypeVar, Type, NamedTuple, Any, get_origin, get_args
import itertools
from collections import namedtuple

from pathlib import Path

import h5py

from structured_qualifiers import IDNumber, Quantity
from record_format import RecordFormat
from table_format import TableFormat
		

class LineDataRecordFormat(RecordFormat):
	mol_id : Annotated[int, IDNumber('RADTRAN ID of molecule')]
	local_iso_id : Annotated[int, IDNumber('Isotope ID in context of parent molecule')]
	nu : Annotated[float, Quantity('Wavenumber of line', 'cm^{-1}')]
	sw : Annotated[float, Quantity('Line intensity at T = 296 K', 'cm^{-1}/(molec.cm^{-2})')]
	a : float
	gamma_air : float
	gamma_self : float
	elower : float
	n_air : float
	delta_air : float
	gp : float
	gpp : float





print(f'{LineDataRecordFormat.metadata('a').as_dict()=}')
print(f'{LineDataRecordFormat}')


			

class LineDataTableFormat(TableFormat):
	record_format : type[RecordFormat] = LineDataRecordFormat
	
	def to_hdf5(self, grp : h5py.Group):
		for i, attr in enumerate(self.__slots__):
			item_path = '/' + attr
			
			if item_path in grp:
				del grp[item_path]
			
			dset = grp.create_dataset(attr, data=getattr(self, attr), dtype=self.record_format.type(attr), maxshape=(None,))
			for k, v in self.record_format.metadata(attr).as_dict().items():
				dset.attrs[k] = v

import numpy as np

a = np.empty((12,50))

print(f'{LineDataTableFormat=}')

ldtf = LineDataTableFormat(*a)
print(f'{ldtf=}')

test_hdf5 = str(Path(__file__).parent / 'test.hdf5')

with h5py.File(test_hdf5, 'a') as f:
	ldtf.to_hdf5(f)