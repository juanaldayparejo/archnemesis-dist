
import numpy as np
from numba import njit

from ._const import NUMBA_CACHE

@njit(cache=NUMBA_CACHE)
def gaussian(
        delta_wn : np.ndarray, 
        alpha_d : float, 
        gamma_l : float
) -> np.ndarray:
    """
    Compute Gaussian profile
    
    ## ARGUMENTS ##
        delta_wn : np.ndarray
            Wavenumber difference from line center
        alpha_d : float
            Doppler width (gaussian HWHM)
        gamma_l : float
            Lorentz width (cauchy-lorentz HWHM) - not used in gaussian profile
    """
    return np.sqrt(np.log(2)/np.pi) / alpha_d * np.exp(- (delta_wn**2 * np.log(2)) / (alpha_d**2))