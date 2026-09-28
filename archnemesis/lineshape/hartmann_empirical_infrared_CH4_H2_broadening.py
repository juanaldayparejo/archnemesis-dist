


import numpy as np
from numba import njit

from ._const import NUMBA_CACHE
from . import voigt

@njit(cache=NUMBA_CACHE)
def hartmann_empirical_infrared_CH4_H2_broadening(
        delta_wn : np.ndarray, 
        alpha_d : float, 
        gamma_l : float
) -> np.ndarray:
    """
    Compute CH4 Voigt lineshape modified for sub-lorentzian line
    wings. Original coefficients recommended by Hartmann (2002)
    """
    chi = np.ones_like(delta_wn)
    abs_delta_wn = np.abs(delta_wn)
    mask_26 = abs_delta_wn < 26
    mask_60 = abs_delta_wn < 60
    
    mask_a = ~mask_26 & mask_60
    
    chi[mask_a] = 8.72*np.exp(-abs_delta_wn[mask_a]/12.0)
    chi[~mask_60] = 0.0684*np.exp(-abs_delta_wn[~mask_60]/ 393.0)
    
    
    return chi*voigt(delta_wn, alpha_d, gamma_l)