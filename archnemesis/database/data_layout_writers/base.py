

from typing import Literal

import h5py

from archnemesis.helpers import h5py_helper
from archnemesis.database.data_layouts.record_layout import RecordLayout
from archnemesis.database.data_layouts.table_layout import TableLayout


class BaseTableWriter(TableLayout):
	record_format : type[RecordLayout] = RecordLayout
	
	def to_hdf5(
			self, 
			grp : h5py.Group, # Group to write the line boradener data to, should probably be something line "/line_data/<mol>/<iso>/broadeners/<broadening_gas>"
			extend : None | Literal['stack'] | int = None
	):
		for i, name in enumerate(self.__slots__):
			#print(f'LineDataTableFormat.to_hdf5(...) {grp=} {i=} {name=} {len(getattr(self, name))=}')
			h5py_helper.ensure_dataset(
				grp, 
				name, 
				attrs=self.record_format.metadata(name).as_dict(),
				extend = extend,
				data=getattr(self, name), dtype=self.record_format.type(name), maxshape=(None,)
			)