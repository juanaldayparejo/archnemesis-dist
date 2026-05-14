import dataclasses as dc

import numpy as np




@dc.dataclass(slots=True)
class PseudoContinuumData:
	s_max                                          : float # Maximum strength of line included in the continuum
	t_cont                                         : float # Temperature the continuum was calculated at (Kelvin)
	p_ref                                          : float # Pressure the continuum was calculated at (atm)
	wn_bin_center                                  : np.ndarray # [N_bins]
	wn_bin_width                                   : np.ndarray # [N_bins]
	line_strength_sum                              : np.ndarray # [N_bins]
	line_strength_weighted_mean_lower_energy_state : np.ndarray # [N_bins]
	line_strength_weighted_gamma_self              : np.ndarray # [N_bins]
	line_strength_weighted_n_self                  : np.ndarray # [N_bins]
	line_strength_weighted_gamma_amb               : np.ndarray # [N_bins, N_broadeners]
	line_strength_weighted_n_amb                   : np.ndarray # [N_bins, N_broadeners]