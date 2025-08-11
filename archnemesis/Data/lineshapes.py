from __future__ import annotations #  for 3.9 compatability

import numpy as np
from scipy.special import wofz, dawsn

def voigt(x: np.ndarray, y: float) -> np.ndarray:
    """
    Compute Voigt profile using Humlicek's algorithm
    Args:
        x: normalized frequency difference (v-v0)/aD
        y: ratio of Lorentz to Doppler widths yL/aD
    """
    z = x + 1j*y
    return wofz(z).real / np.sqrt(np.pi)


def galatry(x: np.ndarray, y: float, beta: float) -> np.ndarray:
    """
    Galatry profile (includes Dicke narrowing)
    Args:
        beta: normalized Dicke narrowing parameter
    """
    z = x + 1j*y
    w = wofz(z)
    d = dawsn(x)
    return (w.real + 2*beta*(x*w.real - d))/np.sqrt(np.pi)


def hartmann(x: np.ndarray, y: float, chi: float, zeta: float) -> np.ndarray:
    """
    Hartmann-Tran profile
    Args:
        chi: velocity-changing collision rate
        zeta: speed-dependence parameter
    """
    z = x + 1j*y
    w = wofz(z)
    return ((1-zeta)*w + zeta*np.exp(-x**2)).real/np.sqrt(np.pi)