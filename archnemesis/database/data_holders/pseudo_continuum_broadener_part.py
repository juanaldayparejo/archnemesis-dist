

import dataclasses as dc

import numpy as np




@dc.dataclass
class PseudoContinuumBroadenerPart:
	name : str # Name of the pseudo-continuum broadener
	
	line_strength_weighted_gamma_amb : np.ndarray
	line_strength_weighted_n_amb : np.ndarray
	