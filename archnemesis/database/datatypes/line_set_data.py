

from typing import NamedTuple

import numpy as np


class LineSetData(NamedTuple):
	s_min        : float # Minimum strength of line included in the line set
	t_ref        : float # Reference temperature line set properties were calculated at
	p_ref        : float # Reference pressure line set properties were calculated at
	req_wn_range : tuple[float,float] # Range of wavenumbers requested (can be less than `min(nu)` or more than `max(nu)`)
	
	mol_id       : np.ndarray # [N_lines]
	local_iso_id : np.ndarray # [N_lines]
	nu           : np.ndarray # [N_lines]
	sw           : np.ndarray # [N_lines]
	a            : np.ndarray # [N_lines]
	elower       : np.ndarray # [N_lines]
	gamma_self   : np.ndarray # [N_lines]
	n_self       : np.ndarray # [N_lines]
	gamma_amb    : np.ndarray # [N_lines, N_broadeners]
	n_amb        : np.ndarray # [N_lines, N_broadeners]
	delta_amb    : np.ndarray # [N_lines, N_broadeners]