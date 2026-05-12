import dataclasses as dc

import numpy as np




@dc.dataclass(slots=True)
class PseudoContinuumData:
	t_cont : float
	wn_bin_center : np.ndarray # [N_bins]
	wn_bin_width : np.ndarray # [N_bins]
	line_strength_sum : np.ndarray # [N_bins]
	line_strength_weighted_mean_lower_energy_state : np.ndarray # [N_bins]
	line_strength_weighted_gamma_self : np.ndarray # [N_bins]
	line_strength_weighted_n_self : np.ndarray # [N_bins]
	line_strength_weighted_gamma_amb : np.ndarray # [N_bins, N_broadeners]
	line_strength_weighted_n_amb : np.ndarray # [N_bins, N_broadeners]