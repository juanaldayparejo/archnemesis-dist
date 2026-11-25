from typing import TYPE_CHECKING, Self, IO

import numpy as np

from ._base import PreRTModelBase
from ..ModelParameter import ModelParameter

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


class Model11(PreRTModelBase):
    """
        Condensing gas without an associated cloud: constant deep abundance XDEEP and,
        where it saturates, the saturated vmr times a relative humidity XRH. The vmr is
        frozen at the cold trap (p < 0.3 atm) once condensation has started.

        ICOND (read from the .apr, VARPARAM(1)), as implemented in subprofretg.f:
            1 -> RH applied only where the gas saturates (above the condensation level)
            0 -> RH applied at every level where p*XDEEP > XRH*psat
        (readapriori.f documents 0 = all levels, 1 = only above condensation, in
        agreement with the code; the comment in subprofretg.f has them swapped.)

        Port of model 11 of NEMESIS (Fortran, radtrancode 91f79ca):
        subprofretg.f (profile), readapriori.f (a priori). SPH / Claude, 17/11/2026.
    """
    id : int = 11

    def __init__(
            self,
            state_vector_start : int,
            n_state_vector_entries : int,
            icond : int,
        ):
        super().__init__(state_vector_start, n_state_vector_entries)
        self.parameters = (
            ModelParameter('XDEEP', slice(0,1), 'Deep gas volume mixing ratio'),
            ModelParameter('XRH', slice(1,2), 'Relative humidity'),
        )
        self.icond = int(icond)
        return

    @classmethod
    def calculate(
            cls,
            atm : "Atmosphere_0",
            atm_profile_type : AtmosphericProfileTypeEnum,
            atm_profile_idx : int | None,
            XDEEP : float,
            XRH : float,
            ICOND : int,
            MakePlot=False
        ) -> tuple["Atmosphere_0", np.ndarray]:
        """
            Returns the updated atmosphere and xmap(2,NP), derivatives of the vmr with
            respect to (XDEEP, XRH) (linear parameters, as in subprofretg.f).
        """
        if atm_profile_type != AtmosphericProfileTypeEnum.GAS_VOLUME_MIXING_RATIO:
            raise ValueError(f'Model id={cls.id} is only defined for gas profiles.')

        NP = atm.NP
        p = atm.P / 101325.0                      # atm
        T = atm.T
        gas_id = int(atm.ID[atm_profile_idx])
        try:
            a, b, c, d = archnemesis.Data.gas_data.svp_coefficients[gas_id]
        except KeyError:
            raise ValueError(f'Model 11 :: no saturation vapour pressure coefficients for gas {gas_id}')

        x1 = np.zeros(NP)
        xmap = np.zeros((2, NP))
        ifla = 0
        for i in range(NP):
            p1 = p[i] * XDEEP
            ps = np.exp(a + b/T[i] + c*T[i] + d*T[i]**2)      # atm
            ph = ps * XRH
            if p1 < ps:
                x1[i] = XDEEP
                xmap[:, i] = (1.0, 0.0)
            else:
                ifla = 1
                x1[i] = ph / p[i]
                xmap[:, i] = (0.0, ps / p[i])
            if ICOND == 0:
                if p1 > ph:
                    x1[i] = ph / p[i]
                    xmap[:, i] = (0.0, ps / p[i])
                else:
                    x1[i] = XDEEP
                    xmap[:, i] = (1.0, 0.0)
            # freeze the vmr at the cold trap once condensation has begun
            if i > 0 and ifla == 1 and p[i] < 0.3 and x1[i] > x1[i-1]:
                x1[i] = x1[i-1]
                xmap[:, i] = xmap[:, i-1]

        vmr = np.array(atm.VMR)
        vmr[:, atm_profile_idx] = x1
        atm.edit_VMR(vmr)
        _lgr.info(f'Model 11 :: gas {gas_id}: XDEEP={XDEEP:.4g} XRH={XRH:.4g} ICOND={ICOND}; '
                  f'condenses: {"yes" if ifla else "no"}; minimum vmr {x1.min():.3g}')
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
        # readapriori.f: xdeep edeep / xrh erh / icond
        ix_0 = ix
        for name in ('xdeep', 'xrh'):
            s = f.readline().split('!')[0].split()
            val, err = float(s[0]), float(s[1])
            if val <= 0.0:
                raise ValueError(f'error in read_apr (Model 11) :: {name} must be > 0')
            x0[ix] = np.log(val)
            lx[ix] = 1
            inum[ix] = 0          # analytical derivatives
            sx[ix, ix] = (err/val)**2
            ix += 1
        icond = int(float(f.readline().split('!')[0].split()[0]))
        varparam[0] = icond

        model_classification = variables.classify_model_type_from_varident(varident, ngas, ndust)
        assert issubclass(cls, model_classification[0]), "Model base class must agree with the classification from Variables_0::classify_model_type_from_varident"

        return cls(ix_0, ix-ix_0, icond)

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
        return cls(ix, 2, int(varparam[0]))

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
        xn_params = self.get_parameter_values_from_state_vector(forward_model.Variables.XN, forward_model.Variables.LX)
        atm, xmap1 = self.calculate(atm, atm_profile_type, atm_profile_idx, xn_params[0], xn_params[1], self.icond)
        # derivatives with respect to log-parameters
        for i, v in enumerate(xn_params):
            if forward_model.Variables.LX[self.state_vector_start+i] == 1:
                xmap1[i, :] *= v
        forward_model.AtmosphereX = atm
        xmap[self.state_vector_slice, ipar, 0:atm.NP] = xmap1
        return
