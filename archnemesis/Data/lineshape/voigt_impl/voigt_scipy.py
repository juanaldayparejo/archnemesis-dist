
import numpy as np
from numba import njit

from .._const import NUMBA_CACHE, SQRT_2log2
from .._scipy_support import voigt_profile

@njit(cache=NUMBA_CACHE)
def voigt_scipy(
        delta_wn : np.ndarray, 
        alpha_d : float, # HWHM of gaussian
        gamma_l : float, # HWHM of cauchy
) -> np.ndarray:
    """
    Compute Voigt profile using Humlicek's algorithm
    
    ## ARGUMENTS ##
        delta_wn : np.ndarray
            Wavenumber difference from line center
        alpha_d : float
            Doppler width (gaussian HWHM)
        gamma_l : float
            Lorentz width (cauchy-lorentz HWHM)
    
    ## CALC ##
    
        sigma : gassian standard deviation
        gamma : lorentz half-width-half-maximum
        
        Gaussian
        --------
        G(dv, sigma) = 1/(sqrt(2*PI) * sigma) EXP(-dv^2 / (2*sigma)^2)
        
        Lorentz
        -------
        L(dv, gamma) = gamma / (PI * (gamma^2 + dv^2))
        
        Voigt
        -----
        V(dv, sigma, gamma) = Re[WOFZ(z)] / (sqrt(2*pi) * sigma)
        
        where:
            z = (dv + gamma * i)/(sqrt(2) * sigma)
        
        
        
        alpha_d : gaussian HWHM = sqrt(2 ln(2)) * sigma
        --->  sigma = aD / sqrt(2 ln(2))
        
        gamma_l : cauchy-lorents HWHM = gamma
    """
    return voigt_profile(delta_wn, alpha_d / SQRT_2log2, gamma_l)