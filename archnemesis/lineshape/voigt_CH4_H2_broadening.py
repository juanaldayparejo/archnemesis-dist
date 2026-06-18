
import numpy as np
from numba import njit

from ._const import NUMBA_CACHE, SQRT_2
from . import voigt

@njit(cache = NUMBA_CACHE)
def voigt_CH4_H2_broadening(
        delta_wn : np.ndarray, 
        alpha_d : float, 
        gamma_l : float
) -> np.ndarray:
    # NOTE: Unsure where the 1/sqrt(2) factor comes from but do not get the same answer as existing LBL tables without it.
    #       it could also be a factor of log(2), they are very similar numbers.
    return voigt(delta_wn, alpha_d/SQRT_2, gamma_l/SQRT_2)