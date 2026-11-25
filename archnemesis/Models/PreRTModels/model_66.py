from typing import TYPE_CHECKING, Self, IO

import numpy as np

from ._base import PreRTModelBase
from ..ModelParameter import ModelParameter

import archnemesis.Data.constants as const
import archnemesis.Data.gas_data
from archnemesis.enum import AtmosphericProfileTypeEnum

from ..log import _lgr  # noqa # Ignore if _lgr is not used


if TYPE_CHECKING:
    from archnemesis.Variables_0 import Variables_0
    from archnemesis.ForwardModel_0 import ForwardModel_0
    from archnemesis.Atmosphere_0 import Atmosphere_0

    nx = 'number of elements in state vector'
    mx = 'synonym for nx'
    mparam = 'the number of parameters a model has'


class Model66(PreRTModelBase):
    """
        Jovian condensing NH3 and two associated clouds.

        Gas: deep abundance below PCLOUD, 'middle' abundance above it (with a
        fractional scale height), limited by a relative humidity once it
        saturates, decreasing with height above the condensation level and frozen
        at the cold trap (p < 0.3 atm).
        Cloud 1 (VARPARAM(1)) peaks at PCLOUD; cloud 2 (VARPARAM(2)) at the
        condensation level PCOND. Both: gaussian (width 0.05 in ln p) below the peak,
        exponential above, optional cut-off at PHAZE1/PHAZE2. Columns renormalised
        to XCDEEP1/XCDEEP2 (as gsetrad.f does in NEMESIS).

        Port of model 66 of NEMESIS (Fortran, radtrancode 91f79ca):
        subprofretg.f, readapriori.f, gsetrad.f. SPH / Claude, 17/11/2026.
    """
    id : int = 66

    def __init__(
            self,
            state_vector_start : int,
            n_state_vector_entries : int,
            jspec1 : int, jspec2 : int,
            phaze1 : float, whaze1 : float,
            phaze2 : float, whaze2 : float,
        ):
        super().__init__(state_vector_start, n_state_vector_entries)

        self.parameters = (
            ModelParameter('XDEEP', slice(0,1), 'Deep gas VMR (p > PCLOUD)'),
            ModelParameter('XMID', slice(1,2), 'Gas VMR just above PCLOUD'),
            ModelParameter('YFSH', slice(2,3), 'Fractional scale height of the gas above PCLOUD'),
            ModelParameter('XRH', slice(3,4), 'Relative humidity at the condensation level'),
            ModelParameter('XSCALE', slice(4,5), 'Scale height of the RH decrease', 'km'),
            ModelParameter('XCDEEP1', slice(5,6), 'Opacity of cloud 1'),
            ModelParameter('XWIDC1', slice(6,7), 'Decay width of cloud 1 above PCLOUD (ln p)'),
            ModelParameter('PCLOUD', slice(7,8), 'Pressure of cloud 1 / gas transition', 'atm'),
            ModelParameter('XCDEEP2', slice(8,9), 'Opacity of cloud 2'),
            ModelParameter('XWIDC2', slice(9,10), 'Decay width of cloud 2 above PCOND (ln p)'),
        )
        self.jspec1, self.jspec2 = jspec1, jspec2
        self.phaze1, self.whaze1 = phaze1, whaze1
        self.phaze2, self.whaze2 = phaze2, whaze2
        self.pcond = None
        return

    @staticmethod
    def _cloud_shape(p, y0, jlim, xwidc, phaze, whaze):
        """Shape of each cloud (subprofretg.f, model 66). jlim: 0-based index of the peak level."""
        NP = len(p)
        y = -np.log(p)
        jj = np.arange(NP)
        q = np.where(jj <= jlim, np.exp(-((y-y0)/0.05)**2), np.exp(-(y-y0)/xwidc))
        khaze = -1
        for j in range(NP-1):
            if p[j] >= phaze and p[j+1] < phaze:
                khaze = j
        if khaze >= 0:
            yhaze = -np.log(phaze)
            xfac = 1.0 - np.exp(-((y-yhaze)/whaze)**2)
            xfac = np.where(jj > khaze, 0.0, xfac)
            q = q * xfac
        return np.maximum(q, 1.0e-36), khaze

    @classmethod
    def calculate(
            cls,
            atm : "Atmosphere_0",
            atm_profile_type : AtmosphericProfileTypeEnum,
            atm_profile_idx : int | None,
            XDEEP, XMID, YFSH, XRH, XSCALE, XCDEEP1, XWIDC1, PCLOUD, XCDEEP2, XWIDC2,
            jspec1, jspec2, phaze1, whaze1, phaze2, whaze2,
            MakePlot=False
        ) -> tuple["Atmosphere_0", np.ndarray, float]:

        if atm_profile_type != AtmosphericProfileTypeEnum.GAS_VOLUME_MIXING_RATIO:
            raise ValueError(f'Model id={cls.id} is only defined for gas profiles.')

        NP = atm.NP
        p = atm.P / 101325.0                      # atm
        if not np.all(np.diff(p) < 0):
            raise ValueError(f'{cls.__name__} :: pressure must decrease with level index')
        T = atm.T
        H = atm.H                                 # m
        scale = const.R * atm.T / (atm.MOLWT * atm.GRAV)   # m

        gas_id = int(atm.ID[atm_profile_idx])
        try:
            a, b, c, d = archnemesis.Data.gas_data.svp_coefficients[gas_id]
        except KeyError:
            raise ValueError(f'Model 66 :: no saturation vapour pressure coefficients for gas {gas_id}')

        yfac = (1.0 - YFSH) / YFSH

        x1 = np.zeros(NP)
        ifla1 = -1          # 0-based index of the first level with p <= PCLOUD
        ifla2 = -1          # 0-based index of the condensation level
        hcond = None
        pcond = None
        for i in range(NP):
            if p[i] > PCLOUD:
                xnow = XDEEP
            else:
                if ifla1 < 0:
                    xnow = XMID
                    ifla1 = i
                else:
                    delh = H[i] - H[i-1]
                    xnow = x1[i-1] * np.exp(-delh * yfac / scale[i])
            p1 = p[i] * xnow
            ps = np.exp(a + b/T[i] + c*T[i] + d*T[i]**2)       # atm
            if p1 < ps and ifla2 < 0:
                x1[i] = xnow
            else:
                if ifla2 < 0:
                    ifla2 = i
                    hcond = H[i]
                    pcond = p[i]
                delh_km = (H[i] - hcond) / 1000.0
                xrh1 = XRH * np.exp(-delh_km / XSCALE)
                x1[i] = xrh1 * ps / p[i]
            # freeze the vmr at the cold trap once condensing
            if i > 0 and ifla2 >= 0 and p[i] < 0.3 and x1[i] > x1[i-1]:
                x1[i] = x1[i-1]

        if ifla1 < 0 or ifla2 < 0:
            raise ValueError(f'Model 66 :: PCLOUD ({PCLOUD} atm) or condensation level not found in the profile '
                             f'(ifla1={ifla1}, ifla2={ifla2})')

        q1, kh1 = cls._cloud_shape(p, -np.log(PCLOUD), ifla1, XWIDC1, phaze1, whaze1)
        q2, kh2 = cls._cloud_shape(p, -np.log(pcond), ifla2, XWIDC2, phaze2, whaze2)

        # gas
        vmr = np.array(atm.VMR)
        vmr[:, atm_profile_idx] = x1
        atm.edit_VMR(vmr)
        # clouds (aerosol index jspec-1), renormalised as gsetrad.f
        dust = np.array(atm.DUST)
        for jspec, q, xc in ((jspec1, q1, XCDEEP1), (jspec2, q2, XCDEEP2)):
            if 0 < jspec <= atm.NDUST:
                dust[:, jspec-1] = q
                atm.DUST_RENORMALISATION[jspec-1] = xc
        atm.edit_DUST(dust)

        _lgr.info(f'Model 66 :: pcloud={PCLOUD:.5g} pcond={pcond:.5g} atm; ifla1,ifla2 (1-based) = {ifla1+1},{ifla2+1}; '
                  f'khaze1,khaze2 (1-based) = {kh1+1 if kh1>=0 else -1},{kh2+1 if kh2>=0 else -1}; '
                  f'clouds {jspec1},{jspec2} renormalised to {XCDEEP1:.4g},{XCDEEP2:.4g}')

        xmap = np.zeros((10, NP))   # numerical derivatives (as NEMESIS for scattering cases)
        return atm, xmap, pcond

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
        # readapriori.f: 10 'value error' lines, then jspec1 jspec2 / phaze1 whaze1 / phaze2 whaze2
        ix_0 = ix
        for k in range(10):
            s = f.readline().split('!')[0].split()
            val, err = float(s[0]), float(s[1])
            if val <= 0.0:
                raise ValueError(f'error in read_apr (Model 66) :: parameter {k+1} must be > 0')
            x0[ix] = np.log(val)
            lx[ix] = 1
            inum[ix] = 1          # numerical derivatives
            sx[ix, ix] = (err/val)**2
            ix += 1
        for k in range(3):
            s = f.readline().split('!')[0].split()
            varparam[2*k] = float(s[0])
            varparam[2*k+1] = float(s[1])

        model_classification = variables.classify_model_type_from_varident(varident, ngas, ndust)
        assert issubclass(cls, model_classification[0]), "Model base class must agree with the classification from Variables_0::classify_model_type_from_varident"

        return cls(ix_0, ix-ix_0, abs(int(varparam[0])), abs(int(varparam[1])),
                   varparam[2], varparam[3], varparam[4], varparam[5])

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
        return cls(ix, 10, abs(int(varparam[0])), abs(int(varparam[1])),
                   varparam[2], varparam[3], varparam[4], varparam[5])

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
        vals = self.get_parameter_values_from_state_vector(forward_model.Variables.XN, forward_model.Variables.LX)
        atm, xmap1, pcond = self.calculate(atm, atm_profile_type, atm_profile_idx, *vals,
                                           self.jspec1, self.jspec2,
                                           self.phaze1, self.whaze1, self.phaze2, self.whaze2)
        self.pcond = pcond
        forward_model.Variables.VARPARAM[ivar, 6] = pcond     # as VARPARAM(IVAR,7) in Fortran
        #State vector elements holding the cloud opacities, needed for the gradients of the renormalisation in calc_tau_dust
        for jspec, ioff in ((self.jspec1, 5), (self.jspec2, 8)):
            if 0 < jspec <= atm.NDUST:
                atm.DUST_RENORMALISATION_IX[jspec-1] = self.state_vector_start + ioff
        forward_model.AtmosphereX = atm
        xmap[self.state_vector_slice, ipar, 0:atm.NP] = xmap1
        return
