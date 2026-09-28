


import numpy as np
import h5py

from archnemesis.helpers import h5py_helper
from archnemesis.database.data_layouts.record_layout import RecordLayout
from archnemesis.database.data_layouts.line_data_record_layout import LineDataRecordLayout

from .base import BaseTableWriter




class LineDataTableWriter(BaseTableWriter):
	record_format : type[RecordLayout] = LineDataRecordLayout
	
	def update_hdf5(
			self, 
			grp : h5py.Group, # Group to update the line data into, should probably be something line "/line_data/<mol>/<iso>"
			wave_attr : str = 'nu', # Attribute that contains the wavenumber of the spectral line
			min_wave_delta_frac = 1E-4 # Minimum fractional difference in wavenumber that spectral lines must have to be considered different lines
	):
	
		dsets = {}
		for i, name in enumerate(self.__slots__):
			dsets[name] = h5py_helper.get_dataset(grp, name, defaults=dict(shape=(0,), dtype=self.record_format.type(name), maxshape=(None,)))
		
		# find duplicate entries, for now assume that a line is duplicated if the fractional difference in wavelength is less than `min_wave_delta_frac`
		mask = np.zeros_like(getattr(self, self.__slots__[0]), dtype=bool)
		wave_values = getattr(self, wave_attr)
		
		print('DEBUG : Updating HDF5, therefore must check to see if data is already present.')
		dset_size = dsets[wave_attr].size
		for i, wave_dset_value in enumerate(dsets[wave_attr]):
			if i%1000 == 0:
				print(f'DEBUG : Updating HDF5 testing if data already present {i}/{dset_size} [{100*i/dset_size:6.2f} %]')
			wave_delta_frac = np.abs((wave_dset_value - wave_values)/wave_values)
			mask |= wave_delta_frac < min_wave_delta_frac # exclude entries that have small differences in wavelength
		
		mask = ~mask # negate mask so it now selects things we want to include (instead of exclude)
		n_old = dsets[wave_attr].size # old number of entries in dataset
		n_new = np.count_nonzero(mask) # number of entries to add to dataset
		print(f'DEBUG : Updating HDF5 {n_old=} {n_new=} {n_old+n_new=}')
		
		for i, name in enumerate(self.__slots__):
			#print(f'LineDataTableFormat.to_hdf5(...) {grp=} {i=} {name=} {len(getattr(self, name))=}')
			dsets[name].resize(n_old+n_new, axis=0)
			dsets[name][n_old:n_old+n_new] = getattr(self,name)[mask]
			
			for k, v in self.record_format.metadata(name).as_dict().items():
				dsets[name].attrs[k] = v


