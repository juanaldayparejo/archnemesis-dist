
import dataclasses as dc
from typing import Iterable

import numpy as np

from archnemesis.database.data_holders.pseudo_continuum_broadener_part import PseudoContinuumBroadenerPart
from archnemesis.database.datatypes.gas_descriptor import RadtranGasDescriptor

@dc.dataclass
class PseudoContinuumDataHolder:
	# source information
	name : str        # Name of source, will result in a "/sources/X" group
	description : str # Description of source, will be the "description" attribute of the "/sources/X" group
	
	# Pseudo-continuum creation parameters
	t_cont : float # Temperature this continuum data was calculated at, Kelvin
	s_max : float  # Maximum line strength included in pseudo-continuum, 'cm^{-1}/(molec.cm^{-2})'
	
	# pseudo-continuum data. 
	mol_id : np.ndarray
	local_iso_id : np.ndarray
	wn_bin_center : np.ndarray
	wn_bin_width : np.ndarray
	line_strength_sum : np.ndarray
	line_strength_weighted_mean_lower_energy_state : np.ndarray
	
	# self broadening
	line_strength_weighted_gamma_self : np.ndarray
	line_strength_weighted_n_self : np.ndarray
	
	# foreign broadening
	broadeners : Iterable[PseudoContinuumBroadenerPart] = tuple()
	
	_rt_gas_descs : None | tuple = None
	
	@property
	def rt_gas_descs(self):
		if self._rt_gas_descs is None:
			u_ids = np.unique(np.array([self.mol_id, self.local_iso_id], dtype=int), axis=1)
			self._rt_gas_descs = tuple(RadtranGasDescriptor(int(gas_id), int(iso_id)) for gas_id, iso_id in u_ids.T)
		return self._rt_gas_descs