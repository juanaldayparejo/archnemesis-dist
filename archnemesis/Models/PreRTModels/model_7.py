
from typing import TYPE_CHECKING, Self, IO

import numpy as np


from ._base import PreRTModelBase
from ..ModelParameter import ModelParameter

import archnemesis.Data.constants as const
from archnemesis.enum import AtmosphericProfileTypeEnum

from ..log import _lgr  # noqa # Ignore if _lgr is not used


if TYPE_CHECKING:
    from archnemesis.Variables_0 import Variables_0
    from archnemesis.ForwardModel_0 import ForwardModel_0
    from archnemesis.Atmosphere_0 import Atmosphere_0

    nx = 'number of elements in state vector'
    mx = 'synonym for nx'
    mparam = 'the number of parameters a model has'


class Model7(PreRTModelBase):
    """
        Variable abundance at a fixed knee pressure level and variable fractional
        scale height, applied both above and below the knee level.

        Port of model 7 of NEMESIS (Fortran): subprofretg.f (profile) and
        readapriori.f (a priori reading).
    """
    id : int = 7

    def __init__(
            self,
            state_vector_start : int,
            n_state_vector_entries : int,
            pknee : float,
            #   Knee pressure (atm)
        ):
        super().__init__(state_vector_start, n_state_vector_entries)

        self.parameters = (
            ModelParameter('ABU_KNEE', slice(0,1), 'Abundance at the knee pressure level'),
            ModelParameter('FSH', slice(1,2), 'Fractional scale height'),
        )

        self.pknee = pknee

        return

    @classmethod
    def calculate(
            cls,
            atm : "Atmosphere_0",
            atm_profile_type : AtmosphericProfileTypeEnum,
            atm_profile_idx : int | None,
            PKNEE : float,
            #   Knee pressure (atm)
            ABU_KNEE : float,
            #   Abundance at the knee level
            FSH : float,
            #   Fractional scale height
            MakePlot=False
        ) -> tuple["Atmosphere_0", np.ndarray]:
        """
            Profile with abundance ABU_KNEE at the model level closest to PKNEE and
            exponential decay (increase) with fractional scale height FSH above (below).

            Follows NEMESIS (Fortran) model 7:
                X(j) = X(j-1) * exp(-dh * xfac / scale(j))   going up from the knee
                X(j) = X(j+1) * exp(-dh * xfac / scale(j))   going down from the knee (dh<0)
            with xfac = (1-FSH)/FSH.

            OUTPUTS :
                atm  :: Updated atmosphere
                xmap(2,npro) :: functional derivatives of the profile wrt (ABU_KNEE, FSH)
                                (linear parameters, as in Model1)
        """

        xfac = (1.0 - FSH) / FSH
        #d(xfac)/d(FSH)
        dxfac = -1.0 / FSH**2.

        NP = atm.NP
        pknee_pa = PKNEE * 101325.   # Pa

        #Level closest to the knee pressure (as in subprofretg.f)
        jfsh = int(np.argmin(np.abs(atm.P - pknee_pa)))
        if jfsh < 1 or jfsh > NP - 2:
            raise ValueError(f'{cls.__name__} :: knee pressure {PKNEE} atm must be within the range of the profile')

        #Scale height (m)
        scale = const.R * atm.T / (atm.MOLWT * atm.GRAV)

        xprof = np.zeros(NP)
        xmap = np.zeros((2,NP))

        xprof[jfsh] = ABU_KNEE
        xmap[0,jfsh] = 1.0

        #Above the knee (increasing index)
        for j in range(jfsh+1, NP):
            delh = atm.H[j] - atm.H[j-1]
            fac = np.exp(-delh * xfac / scale[j])
            xprof[j] = xprof[j-1] * fac
            xmap[0,j] = xmap[0,j-1] * fac
            xmap[1,j] = (-delh / scale[j]) * dxfac * xprof[j-1] * fac + xmap[1,j-1] * fac
            if xprof[j] < 1.0e-36:
                xprof[j] = 1.0e-36

        #Below the knee (decreasing index)
        for j in range(jfsh-1, -1, -1):
            delh = atm.H[j] - atm.H[j+1]
            fac = np.exp(-delh * xfac / scale[j])
            xprof[j] = xprof[j+1] * fac
            xmap[0,j] = xmap[0,j+1] * fac
            xmap[1,j] = (-delh / scale[j]) * dxfac * xprof[j+1] * fac + xmap[1,j+1] * fac
            if xprof[j] > 1.0e10:
                xprof[j] = 1.0e10

        #Updating atmosphere class
        if atm_profile_type == AtmosphericProfileTypeEnum.GAS_VOLUME_MIXING_RATIO:
            tmp = np.array(atm.VMR)
            tmp[:,atm_profile_idx] = xprof
            atm.edit_VMR(tmp)
        elif atm_profile_type == AtmosphericProfileTypeEnum.TEMPERATURE:
            atm.edit_T(xprof)
        elif atm_profile_type == AtmosphericProfileTypeEnum.AEROSOL_DENSITY:
            tmp = np.array(atm.DUST)
            tmp[:,atm_profile_idx] = xprof
            atm.edit_DUST(tmp)
        elif atm_profile_type == AtmosphericProfileTypeEnum.PARA_H2_FRACTION:
            atm.PARAH2 = xprof
        elif atm_profile_type == AtmosphericProfileTypeEnum.FRACTIONAL_CLOUD_COVERAGE:
            atm.FRAC = xprof
        else:
            raise ValueError(f'{cls.__name__} id {cls.id} has unknown atmospheric profile type {atm_profile_type}')

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
        ix_0 = ix
        #******** profile held as value at a knee pressure and fractional scale height
        #Read: pknee / xknee eknee / xfsh efsh   (as in readapriori.f)

        s = f.readline().split()
        pknee = float(s[0])
        varparam[0] = pknee

        s = f.readline().split()
        xknee = float(s[0])
        eknee = float(s[1])

        if varident[0] == 0:  #Temperature
            x0[ix] = xknee
            err = eknee
        else:  #VMR, para-H2, fcloud or cloud
            if xknee > 0.0:
                x0[ix] = np.log(xknee)
                lx[ix] = 1
                inum[ix] = 0  #Analytical derivatives
            else:
                raise ValueError('error in read_apr (Model 7) :: xknee must be > 0 if it applies to any parameter but temperature')
            err = eknee / xknee
        sx[ix,ix] = err**2.
        ix += 1

        s = f.readline().split()
        fsh = float(s[0])
        efsh = float(s[1])
        if fsh > 0.0:
            x0[ix] = np.log(fsh)
            lx[ix] = 1
            inum[ix] = 0  #Analytical derivatives
            sx[ix,ix] = (efsh / fsh)**2.
        else:
            raise ValueError('error in read_apr (Model 7) :: fsh must be > 0')
        ix += 1

        model_classification = variables.classify_model_type_from_varident(varident, ngas, ndust)
        assert issubclass(cls, model_classification[0]), "Model base class must agree with the classification from Variables_0::classify_model_type_from_varident"

        return cls(ix_0, ix-ix_0, pknee)

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
        ix_0 = ix
        ix = ix + 2
        pknee = varparam[0]
        return cls(ix_0, ix-ix_0, pknee)

    def calculate_from_subprofretg(
            self,
            forward_model : "ForwardModel_0",
            ix : int,
            ipar : int,
            ivar : int,
            xmap : np.ndarray,
        ) -> None:
        #Model 7. Abundance at a fixed knee pressure + fractional scale height above and below
        #***************************************************************
        atm = forward_model.AtmosphereX
        atm_profile_type, atm_profile_idx = atm.ipar_to_atm_profile_type(ipar)

        xn_params = self.get_parameter_values_from_state_vector(forward_model.Variables.XN, forward_model.Variables.LX)
        xn_params_1d = np.concatenate([np.atleast_1d(v) for v in xn_params])

        atm, xmap1 = self.calculate(
            atm,
            atm_profile_type,
            atm_profile_idx,
            self.pknee,
            xn_params[0],
            xn_params[1],
        )

        #Derivatives wrt log-parameters need the parameter value
        for i in range(self.n_state_vector_entries):
            if forward_model.Variables.LX[self.state_vector_start+i] == 1:
                xmap1[i,:] = xmap1[i,:] * xn_params_1d[i]

        forward_model.AtmosphereX = atm
        xmap[self.state_vector_slice, ipar, 0:atm.NP] = xmap1

        return
