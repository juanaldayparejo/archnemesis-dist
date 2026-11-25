"""
Tests for the refractive index used by model 444 (Models/PreRTModels/model_444.py::refind_nemesis).

NEMESIS-Fortran (gsetrad.f, model 444 block) interpolates ln(n_imag) with VERINT onto the
calculation grid, in the spectral units of the calculation (ISPACE), and obtains n_real with
kk_new_sub.f applied in wavenumber. The reference below is a literal port of that Fortran code.
"""
import numpy as np

from archnemesis.enum import WaveUnitEnum
from archnemesis.Models.PreRTModels.model_444 import refind_nemesis, _verint


class _Scatter:
    def __init__(self, wave, ispace):
        self.WAVE = np.asarray(wave, float)
        self.ISPACE = ispace


def _kk_fortran(vi, k, vm, nm):
    """Literal port of radtran/scatter/kk_new_sub.f (wavenumber grid)."""
    rev = vi[0] > vi[-1]
    va, ka = (vi[::-1], k[::-1]) if rev else (vi, k)
    km = _verint(va, ka, np.array([vm]))[0]
    na = np.zeros(len(va))
    for i, v in enumerate(va):
        alpha = va**2 - v**2
        beta = va**2 - vm**2
        ok = (alpha != 0) & (beta != 0)
        y = np.zeros(len(va))
        y[ok] = (ka[ok]*va[ok] - ka[i]*v)/alpha[ok] - (ka[ok]*va[ok] - km*vm)/beta[ok]
        na[i] = nm + 2.0/np.pi*np.sum(0.5*(y[1:] + y[:-1])*np.diff(va))
    return na[::-1] if rev else na


# Synthetic 444 nodes (wavelength, um) with strongly varying n_imag
NODES_UM = np.array([0.25, 0.35, 0.5, 0.7, 1.0, 1.2])
NIMAG = np.array([3e-2, 1e-2, 2e-3, 1e-4, 5e-5, 3e-4])
WAVE_REF_UM, NREAL_REF = 0.5, 1.33
GRID_UM = np.linspace(0.2, 1.0, 81)


def test_refind_nemesis_matches_fortran_wavelength():
    wavel, nr, ni = refind_nemesis(_Scatter(GRID_UM, WaveUnitEnum.Wavelength_um),
                                   NODES_UM, np.log(NIMAG), WAVE_REF_UM, NREAL_REF)
    ni_f = np.exp(_verint(NODES_UM, np.log(NIMAG), GRID_UM))
    nr_f = _kk_fortran(1e4/GRID_UM[::-1], ni_f[::-1], 1e4/WAVE_REF_UM, NREAL_REF)[::-1]
    assert np.allclose(wavel, GRID_UM)
    assert np.allclose(ni, ni_f, rtol=1e-10)
    assert np.max(np.abs(nr - nr_f)) < 1e-3


def test_refind_nemesis_matches_fortran_wavenumber():
    nodes_cm, nimag_cm = 1e4/NODES_UM[::-1], NIMAG[::-1]
    grid_cm = 1e4/GRID_UM[::-1]
    wavel, nr, ni = refind_nemesis(_Scatter(grid_cm, WaveUnitEnum.Wavenumber_cm),
                                   nodes_cm, np.log(nimag_cm), 1e4/WAVE_REF_UM, NREAL_REF)
    ni_f = np.exp(_verint(nodes_cm, np.log(nimag_cm), grid_cm))
    nr_f = _kk_fortran(grid_cm, ni_f, 1e4/WAVE_REF_UM, NREAL_REF)
    # Output is on a wavelength grid (um), ascending, as expected by Scatter_0.makephase
    assert np.allclose(wavel, 1e4/grid_cm[::-1])
    assert np.allclose(ni, ni_f[::-1], rtol=1e-10)
    assert np.max(np.abs(nr - nr_f[::-1])) < 1e-3


def test_refind_nemesis_single_nimag_gives_constant_nreal():
    wavel, nr, ni = refind_nemesis(_Scatter(GRID_UM, WaveUnitEnum.Wavelength_um),
                                   NODES_UM[:1], np.log(NIMAG[:1]), WAVE_REF_UM, NREAL_REF)
    assert np.allclose(nr, NREAL_REF)
    assert np.allclose(ni, NIMAG[0])
