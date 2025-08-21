from __future__ import annotations #  for 3.9 compatability

import numpy as np
from scipy.special import voigt_profile#, wofz, dawsn

SQRT_2 = np.sqrt(2)
SQRT_PI = np.sqrt(np.pi)
SQRT_2PI = SQRT_2 * SQRT_PI
SQRT_2log2 = np.sqrt(2*np.log(2))
SQRT_log2 = np.sqrt(np.log(2))


def voigt(
        delta_wn : np.ndarray, 
        alpha_d : float, 
        gamma_l : float
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
    #sigma_sqrt_2 = alpha_d / SQRT_log2
    #z = (delta_wn/sigma_sqrt_2) + 1j*(gamma_l/sigma_sqrt_2)
    #return wofz(z).real / (SQRT_PI * sigma_sqrt_2)
    
    #sigma = alpha_d / SQRT_2log2
    #z = (delta_wn + 1j*gamma_l)/(SQRT_2*sigma)
    #return wofz(z).real / (SQRT_2PI * sigma)
    
    sigma = alpha_d / SQRT_2log2
    return voigt_profile(delta_wn, sigma, gamma_l)



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
    
    mask_26 = delta_wn < 26
    mask_60 = delta_wn < 60
    
    mask_a = ~mask_26 & mask_60
    
    chi[mask_a] = 8.72*np.exp(-np.abs(delta_wn[mask_a])/12.0)
    chi[~mask_60] = 0.0684*np.exp(-np.abs(delta_wn[~mask_60])/ 393.0)
    
    return chi*voigt(delta_wn, alpha_d, gamma_l)