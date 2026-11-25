from typing import TYPE_CHECKING, Self, IO

import numpy as np

from ._base import PreRTModelBase
from ..ModelParameter import ModelParameter

from archnemesis.enum import AtmosphericProfileTypeEnum

from ..log import _lgr  # noqa # Ignore if _lgr is not used


if TYPE_CHECKING:
    from archnemesis.Variables_0 import Variables_0
    from archnemesis.ForwardModel_0 import ForwardModel_0
    from archnemesis.Atmosphere_0 import Atmosphere_0

    nx = 'number of elements in state vector'
    mx = 'synonym for nx'
    mparam = 'the number of parameters a model has'


class Model54(PreRTModelBase):
    """
        Cloud profile based on the Galileo probe nephelometer main cloud: gaussian
        (in log pressure) below the peak pressure and exponential above it, with a
        cut-off above a fixed 'haze' pressure. The column is renormalised to the
        retrieved opacity (as gsetrad.f does in NEMESIS).

        Port of model 54 of NEMESIS (Fortran, radtrancode 91f79ca):
        subprofretg.f (profile), readapriori.f (a priori), gsetrad.f (renormalisation).
        SPH / Claude, 17/11/2026.
    """
    id : int = 54

    def __init__(
            self,
            state_vector_start : int,
            n_state_vector_entries : int,
            phaze : float,
            #   Haze cut-off pressure (atm), VARPARAM(1)
            whaze : float,
            #   Width of the cut-off (log pressure), VARPARAM(2)
        ):
        super().__init__(state_vector_start, n_state_vector_entries)

        self.parameters = (
            ModelParameter('XDEEP', slice(0,1), 'Integrated cloud opacity (at the normalising wavelength)'),
            ModelParameter('PKNEE', slice(1,2), 'Pressure of the cloud peak', 'atm'),
            ModelParameter('XWID', slice(2,3), 'Decay width above the peak (log pressure)'),
            ModelParameter('XWID1', slice(3,4), 'Gaussian width below the peak (log pressure)'),
        )
        self.phaze = phaze
        self.whaze = whaze
        return

    @classmethod
    def calculate(
            cls,
            atm : "Atmosphere_0",
            atm_profile_type : AtmosphericProfileTypeEnum,
            atm_profile_idx : int | None,
            XDEEP : float,
            PKNEE : float,
            XWID : float,
            XWID1 : float,
            PHAZE : float,
            WHAZE : float,
            MakePlot=False
        ) -> tuple["Atmosphere_0", np.ndarray]:
        """
            Shape (particles per gram, arbitrary scale), Fortran index J -> python j=J-1:
                y = -ln(p);  y0 = -ln(PKNEE)
                q = exp(-((y-y0)/XWID1)**2)   for j <= k   (k: P[k] >= PKNEE > P[k+1])
                q = exp(-(y-y0)/XWID)         for j >  k
                q *= 1 - exp(-((y-yhaze)/WHAZE)**2);  q = 0 for j > khaze
            NOTE: as in the Fortran, yhaze = +ln(PHAZE) (sign opposite to y). With
            the usual values this makes the taper negligible and leaves a sharp
            cut-off at PHAZE. Kept as is to reproduce NEMESIS.
            The column is then renormalised to XDEEP through DUST_RENORMALISATION.
        """
        if atm_profile_type != AtmosphericProfileTypeEnum.AEROSOL_DENSITY:
            raise ValueError(f'Model id={cls.id} is only defined for aerosol profiles.')

        p = atm.P / 101325.0          # atm
        NP = atm.NP
        if not np.all(np.diff(p) < 0):
            raise ValueError(f'{cls.__name__} :: pressure must decrease with level index')

        y0 = -np.log(PKNEE)
        yhaze = np.log(PHAZE)         # sign as in subprofretg.f (see docstring)

        k = -1
        khaze = -1
        for j in range(NP-1):
            if p[j] >= PKNEE and p[j+1] < PKNEE:
                k = j
            if p[j] >= PHAZE and p[j+1] < PHAZE:
                khaze = j
        if k < 0:
            raise ValueError(f'{cls.__name__} :: cannot find the knee pressure {PKNEE} atm in the profile')

        y = -np.log(p)
        jj = np.arange(NP)
        q = np.where(jj <= k, np.exp(-((y-y0)/XWID1)**2), np.exp(-(y-y0)/XWID))
        xfac = 1.0 - np.exp(-((y-yhaze)/WHAZE)**2)
        xfac = np.where(jj > khaze, 0.0, xfac)
        q = q * xfac
        q = np.maximum(q, 1.0e-36)

        tmp = np.array(atm.DUST)
        if tmp.ndim == 1:
            tmp = tmp[:, None]
        tmp[:, atm_profile_idx] = q
        atm.edit_DUST(tmp)
        atm.DUST_RENORMALISATION[atm_profile_idx] = XDEEP

        _lgr.info(f'Model 54 :: aerosol {atm_profile_idx+1}: PKNEE={PKNEE:.4g} atm (level {k}), '
                  f'PHAZE={PHAZE:.3g} atm (level {khaze}), column renormalised to {XDEEP:.4g}')

        # Analytical derivatives not implemented (NEMESIS also computes them
        # numerically for scattering cases).
        xmap = np.zeros((4, NP))
        return atm, xmap

    @classmethod
    def from_apr_to_state_vector(
            cls,
            variables : "Variables_0",
            f : IO,
            varident : np.ndarray[[3],int],
            varparam : np.ndarray[["mparam"],float],
            ix : int,
            lx : np.ndarray[["mx"],int],
            x0 : np.ndarray[["mx"],float],
            sx : np.ndarray[["mx","mx"],float],
            inum : np.ndarray[["mx"],int],
            npro : int,
            ngas : int,
            ndust : int,
            nlocations : int,
            runname : str,
            sxminfac : float,
        ) -> Self:
        # readapriori.f: xdeep edeep / pknee eknee / xwid ewid / xwid1 ewid1 / phaze whaze
        ix_0 = ix
        names = ('xdeep', 'pknee', 'xwid', 'xwid1')
        for name in names:
            s = f.readline().split('!')[0].split()
            val, err = float(s[0]), float(s[1])
            if val <= 0.0:
                raise ValueError(f'error in read_apr (Model 54) :: {name} must be > 0')
            x0[ix] = np.log(val)
            lx[ix] = 1
            inum[ix] = 1          # numerical derivatives
            sx[ix, ix] = (err/val)**2
            ix += 1
        s = f.readline().split('!')[0].split()
        varparam[0] = float(s[0])
        varparam[1] = float(s[1])

        model_classification = variables.classify_model_type_from_varident(varident, ngas, ndust)
        assert issubclass(cls, model_classification[0]), "Model base class must agree with the classification from Variables_0::classify_model_type_from_varident"

        return cls(ix_0, ix-ix_0, varparam[0], varparam[1])

    @classmethod
    def from_bookmark(
            cls,
            variables : "Variables_0",
            varident : np.ndarray[[3],int],
            varparam : np.ndarray[["mparam"],float],
            ix : int,
            npro : int,
            ngas : int,
            ndust : int,
            nlocations : int,
        ) -> Self:
        return cls(ix, 4, varparam[0], varparam[1])

    def calculate_from_subprofretg(
            self,
            forward_model : "ForwardModel_0",
            ix : int,
            ipar : int,
            ivar : int,
            xmap : np.ndarray,
        ) -> None:
        atm = forward_model.AtmosphereX
        atm_profile_type, atm_profile_idx = atm.ipar_to_atm_profile_type(ipar)
        xdeep, pknee, xwid, xwid1 = self.get_parameter_values_from_state_vector(
            forward_model.Variables.XN, forward_model.Variables.LX)
        atm, xmap1 = self.calculate(atm, atm_profile_type, atm_profile_idx,
                                    xdeep, pknee, xwid, xwid1, self.phaze, self.whaze)
        #State vector element holding the cloud opacity, needed for the gradients of the renormalisation in calc_tau_dust
        atm.DUST_RENORMALISATION_IX[atm_profile_idx] = self.state_vector_start
        forward_model.AtmosphereX = atm
        xmap[self.state_vector_slice, ipar, 0:atm.NP] = xmap1
        return
