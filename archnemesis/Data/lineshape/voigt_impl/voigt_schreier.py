
import numpy as np
from numba import njit

from .._const import (
	NUMBA_CACHE, 
	SQRT_2, 
	SQRT_log2, 
	INV_SQRT_2PI, 
	INV_SQRT_PI, 
	A24, 
	L24,
)


@njit(cache=NUMBA_CACHE)
def complex_err_fn_weideman_24a(z_r, z_i):
    """
    Complex error function --- J.A.C. Weideman: SIAM J. Num. Anal. 31 1497-1518 (1994); equation (38.I) and table I.
    """
    # In this case N = 24
    # L = 2^(1/4) * N^(1/2)
    # L = L24 from constants
    #print(f'{L24=}')
    
    # L + i*z
    lp_iz_r = L24 - z_i
    lp_iz_i = z_r
    #print(f'{lp_iz_r=} {lp_iz_i=}')
    
    # L - i*z
    lm_iz_r = L24 + z_i
    lm_iz_i = -z_r
    #print(f'{lm_iz_r=} {lm_iz_i=}')
    
    # 1.0 / (L - i*z)
    mag_lm_iz = lm_iz_r*lm_iz_r + lm_iz_i*lm_iz_i
    #print(f'{mag_lm_iz=}')
    inv_lm_iz_r = lm_iz_r / mag_lm_iz
    inv_lm_iz_i = -1*lm_iz_i / mag_lm_iz
    #print(f'{inv_lm_iz_r=} {inv_lm_iz_i=}')
    
    # Z = (L + i*z) / (L - i*z)
    # (a + bi)*(x + yi) = (a*x - b*y) + (a*y + b*x)i
    Z_r = lp_iz_r * inv_lm_iz_r - lp_iz_i * inv_lm_iz_i
    Z_i = lp_iz_r * inv_lm_iz_i + lp_iz_i * inv_lm_iz_r
    #print(f'{Z_r=} {Z_i=}')
    
    #              1.0                   2.0         N-1 /               \
    # w = --------------------  + ----------------  SUM | a_{n+1} * Z^{n} |
    #      PI^(1/2) (L - i*z)       (L - i*z)^(2)    n=0 \               /
    
    # HORNER (IMPORTANT for speed + stability)
    # poly = a_1*Z^0 + a_2*Z^1 + a_3*Z^2 + ... + a_N*Z^{n-1}
    # poly = a_1 + Z(a_2 + Z( a_3+ Z( ... Z(a_{N-1} + Z*a_N) ... ) ) ) 
    
    polynom_r = A24[-1]
    polynom_i = 0.0
    for i in range(A24.size-2,0,-1):
        #print(f'{i=} {A24[i]=} {polynom_r} {polynom_i}')
        Z_poly_r = polynom_r * Z_r - polynom_i * Z_i
        Z_poly_i = polynom_r * Z_i + polynom_i * Z_r
        polynom_r = Z_poly_r + A24[i]
        polynom_i = Z_poly_i
    
    #print(f'{polynom_r} {polynom_i}')
    
    # w = (INV_SQRT_PI  +  2.0 * polynom * inv_lmi_z)  *  inv_lmi_z
    # polynom * inv_lmi_z
    x_r = polynom_r * inv_lm_iz_r - polynom_i * inv_lm_iz_i
    x_i = polynom_r * inv_lm_iz_i + polynom_i * inv_lm_iz_r
    # (INV_SQRT_PI  +  2.0 * polynom * inv_lmi_z)
    x_r = INV_SQRT_PI + 2.0 * x_r
    x_i = 2.0 * x_i
    # (INV_SQRT_PI  +  2.0 * polynom * inv_lmi_z)  *  inv_lmi_z
    w_r = x_r * inv_lm_iz_r - x_i * inv_lm_iz_i
    w_i = x_r * inv_lm_iz_i + x_i * inv_lm_iz_r
    
    #print(f'{w_r=} {w_i=}\n')
    
    return (w_r, w_i)

@njit(cache=NUMBA_CACHE)
def voigt_schreier(
        delta_wn : float, 
        alpha_d : float, 
        gamma_l : float
    ) -> float:
    """
    Compute Voigt profile using Weideman's algorithm

    Computations are based on:
    Schreier, F. (2018). Comments on the Voigt function implementation in the Astropy and SpectraPlot. com packages. Journal of Quantitative Spectroscopy and Radiative Transfer, 213, 13-16.

    ## ARGUMENTS ##
        delta_wn : float
            Wavenumber difference from line center
        alpha_d : float
            Doppler width (gaussian HWHM)
        gamma_l : float
            Lorentz width (cauchy-lorentz HWHM)
    """

    # precompute scaling (scalar)
    scale = SQRT_log2 / alpha_d

    y = gamma_l * scale

    x = delta_wn * scale

    w_r, w_i = complex_err_fn_weideman_24a(x, y)

    return w_r * scale * INV_SQRT_2PI * SQRT_2

@njit(cache=NUMBA_CACHE)
def voigt_schreier_arr(
        delta_wn : np.ndarray, 
        alpha_d : float, 
        gamma_l : float,
        out : np.ndarray,
    ) -> float:
    """
    Compute Voigt profile using Weideman's algorithm

    Computations are based on:
    Schreier, F. (2018). Comments on the Voigt function implementation in the Astropy and SpectraPlot. com packages. Journal of Quantitative Spectroscopy and Radiative Transfer, 213, 13-16.

    ## ARGUMENTS ##
        delta_wn : float
            Wavenumber difference from line center
        alpha_d : float
            Doppler width (gaussian HWHM)
        gamma_l : float
            Lorentz width (cauchy-lorentz HWHM)
    """

    # precompute scaling (scalar)
    scale = SQRT_log2 / alpha_d

    y = gamma_l * scale
    
    for i in range(delta_wn.size):

        x = delta_wn * scale

        w_r, w_i = complex_err_fn_weideman_24a(x, y)

        out[i] = w_r * scale * INV_SQRT_2PI * SQRT_2