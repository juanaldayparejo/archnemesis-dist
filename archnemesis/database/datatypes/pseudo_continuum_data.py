import dataclasses as dc

import numpy as np




@dc.dataclass(slots=True)
class PseudoContinuumData:
	s_max                                          : float # Maximum strength of line included in the continuum
	t_cont                                         : float # Temperature the continuum was calculated at (Kelvin)
	p_cont                                         : float # Pressure the continuum was calculated at (atm)
	req_wn_range                                   : tuple[float,float] # Range of wavenumbers requested (can be less than `min(wn_bin_center)` or more than `max(wn_bin_center)`)

	wn_bin_center                                  : np.ndarray # [N_bins]
	wn_bin_width                                   : np.ndarray # [N_bins]
	line_strength_sum                              : np.ndarray # [N_bins]
	line_strength_weighted_mean_lower_energy_state : np.ndarray # [N_bins]
	line_strength_weighted_gamma_self              : np.ndarray # [N_bins]
	line_strength_weighted_n_self                  : np.ndarray # [N_bins]
	line_strength_weighted_gamma_amb               : np.ndarray # [N_bins, N_broadeners]
	line_strength_weighted_n_amb                   : np.ndarray # [N_bins, N_broadeners]