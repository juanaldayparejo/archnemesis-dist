

import dataclasses as dc

import numpy as np




@dc.dataclass
class PseudoContinuumBroadenerPart:
	name : str # Name of the pseudo-continuum broadener
	
	line_strength_weighted_gamma_amb : np.ndarray
	line_strength_weighted_n_amb : np.ndarray
	
	t_ref : float = 296 # Reference temperature at which data was calculated
	t_unit : str = 'Kelvin' # Unit of reference temperature
	p_ref : float = 1 # Reference pressure
	p_unit : str = 'atm' # Unit of reference pressure