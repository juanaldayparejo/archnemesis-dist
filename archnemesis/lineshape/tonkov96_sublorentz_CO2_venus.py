

import numpy as np
from numba import njit

from ._const import NUMBA_CACHE
from . import voigt

@njit(cache=NUMBA_CACHE)
def tonkov96_sublorentz_CO2_venus(
        delta_wn : np.ndarray, 
        alpha_d : float, 
        gamma_l : float
) -> np.ndarray:
    """
    Compute a CO2 Voigt lineshape modified for sub-lorentzian line wings and line mixing
    for Venus near-infrared windows

    Here we use the chi values from: 
        Tonkov M.V. et al., 1996, Measurements and empirical
        modeling of pure CO2 absorption in the 2.3um region at
        room temperature: far wings, allowed and collision-
        induced bands, Applied Optics, 35, 24, 4863-4870
    
    ## ARGUMENTS ##
        delta_wn : np.ndarray
            Wavenumber difference from line center
        alpha_d : float
            Doppler width (gaussian HWHM)
        gamma_l : float
            Lorentz width (cauchy-lorentz HWHM)
    """

    #Getting the chi values from Tonkov+96
    chi = np.zeros_like(delta_wn)

    mask = np.abs(delta_wn) < 3.
    chi[mask] = 1.

    mask = (np.abs(delta_wn) >= 3.) & (np.abs(delta_wn) < 150.)
    chi[mask] = 1.084 * np.exp(-0.027 * (np.abs(delta_wn[mask])))

    mask = (np.abs(delta_wn) >= 150.) & (np.abs(delta_wn) < 300.)
    chi[mask] = 0.208 * np.exp(-0.016 * (np.abs(delta_wn[mask])))

    mask = (np.abs(delta_wn) >= 300.)
    chi[mask] = 0.025 * np.exp(-0.009 * (np.abs(delta_wn[mask])))

    return chi*voigt(delta_wn, alpha_d, gamma_l)