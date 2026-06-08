

import numpy as np
#from scipy.special import (
    #voigt_profile, 
    #wofz, 
    #dawsn
#)


# Borrowed from https://github.com/scikit-hep/numba-stats/blob/main/src/numba_stats/_special.py
#
# numba currently does not support scipy, so we cannot access
# scipy.stats.norm.ppf and scipy.stats.poisson.cdf in a JIT'ed
# function. As a workaround, we wrap special functions from
# scipy to implement the needed functions here.
from typing import Any

from numba import njit
from numba.extending import get_cython_function_address
from numba.types import WrapperAddressProtocol, float64


def get(name: str, signature: Any) -> Any:
    # create new function object with correct signature that numba can call
    from scipy.special import cython_special

    # scipy-1.12 started to provide fused versions for some special functions
    if name in {"betainc", "stdtr", "stdtrit"}:
        fuse_name = f"__pyx_fuse_0{name}"
    else:
        fuse_name = f"__pyx_fuse_1{name}"
    if fuse_name not in cython_special.__pyx_capi__:
        fuse_name = name

    addr = get_cython_function_address("scipy.special.cython_special", fuse_name)

    # dynamically create type that inherits from WrapperAddressProtocol
    cls = type(
        name,
        (WrapperAddressProtocol,),
        {"__wrapper_address__": lambda self: addr, "signature": lambda self: signature},
    )
    return cls()

# VOIGT PROFILE ##########################################################
voigt_profile = get("voigt_profile", float64(float64, float64, float64)) #
# ARGUMENTS -------------------------------------------------------------#
#   x - Value to calculate probability for                               #
#   sigma - Standard deviation of normal distribution part               #
#   gamma - Half-width at Half-maximum of Cauchy distribution part       #
##########################################################################

SQRT_2 = np.sqrt(2)
SQRT_PI = np.sqrt(np.pi)
SQRT_2PI = SQRT_2 * SQRT_PI
SQRT_2log2 = np.sqrt(2*np.log(2))
SQRT_log2 = np.sqrt(np.log(2))



@njit
def voigt(
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

@njit
def voigt_ch4_H2_ambient(
        delta_wn : np.ndarray, 
        alpha_d : float, 
        gamma_l : float
    ) -> np.ndarray:
    
    
    # NOTE: Unsure where the log(2) factor comes from but do not get the same answer as existing LBL tables without it.
    #       it could also be a factor of 1/sqrt(2), they are very similar numbers.
    #return voigt_profile(delta_wn, alpha_d, np.log(2)*gamma_l) 

    # NOTE: Unsure where the 1/sqrt(2) factor comes from but do not get the same answer as existing LBL tables without it.
    #       it could also be a factor of log(2), they are very similar numbers.
    return voigt_profile(delta_wn, alpha_d/SQRT_2, gamma_l/SQRT_2)



@njit
def hartmann_empirical_infrared_ch4_h2_broadening(
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

@njit
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

@njit
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
























