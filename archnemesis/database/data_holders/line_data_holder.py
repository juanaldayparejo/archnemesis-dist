
import dataclasses as dc
from typing import Iterable

import numpy as np

from archnemesis.database.data_holders.line_broadener_holder import LineBroadenerHolder
from archnemesis.database.datatypes.gas_descriptor import RadtranGasDescriptor

@dc.dataclass
class LineDataHolder:
	# source information
	name : str
	description : str
	
	# spectral line data
	mol_id : np.ndarray
	local_iso_id : np.ndarray
	nu : np.ndarray
	sw : np.ndarray
	a : np.ndarray
	elower : np.ndarray
	
	# self broadening
	gamma_self : None | np.ndarray = None
	n_self : None | np.ndarray = None
	
	# foreign broadening
	broadeners : Iterable[LineBroadenerHolder] = tuple()
	
	_rt_gas_descs : None | tuple = None
	
	def __post_init__(self):
		if self.gamma_self is None:
			self.gamma_self = np.ones_like(self.nu, dtype=float)
		
		if self.n_self is None:
			self.n_self = np.zeros_like(self.nu, dtype=float)
		return
	
	@property
	def rt_gas_descs(self):
		if self._rt_gas_descs is None:
			u_ids = np.unique(np.array([self.mol_id, self.local_iso_id], dtype=int), axis=1)
			self._rt_gas_descs = tuple(RadtranGasDescriptor(int(gas_id), int(iso_id)) for gas_id, iso_id in u_ids.T)
		return self._rt_gas_descs