
import numpy as np
from numba import njit

from ._const import NUMBA_CACHE

@njit(cache=NUMBA_CACHE)
def lorentz(
        delta_wn : np.ndarray, 
        alpha_d : float,
        gamma_l : float
) -> np.ndarray:
    """
    Compute Lorentz profile
    
    ## ARGUMENTS ##
        delta_wn : np.ndarray
            Wavenumber difference from line center
        alpha_d : float
            Doppler width (gaussian HWHM) - not used in lorentz profile
        gamma_l : float
            Lorentz width (cauchy-lorentz HWHM)
    """
    return gamma_l / (np.pi * (gamma_l**2 + delta_wn**2))