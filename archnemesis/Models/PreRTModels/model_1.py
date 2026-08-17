

from typing import TYPE_CHECKING, Self, IO, ClassVar
import dataclasses as dc

import numpy as np


from ._base import PreRTModelBase
from ..param import StateParam, ConstParam, VarParam

import archnemesis.Data.constants as const
from archnemesis.enum import AtmosphericProfileTypeEnum, ArchNemesisFileTypeEnum

from ..log import _lgr  # noqa # Ignore if _lgr is not used


if TYPE_CHECKING:
    from archnemesis.Variables_0 import Variables_0
    from archnemesis.Atmosphere_0 import Atmosphere_0

    nx = 'number of elements in state vector'
    m = 'an undetermined number, but probably less than "nx"'
    mx = 'synonym for nx'
    mparam = 'the number of parameters a model has'
    nparam = 'the number of parameters a model has'
    NCONV = 'number of spectral bins'
    NGEOM = 'number of geometries'
    NX = 'number of elements in state vector'
    NDEGREE = 'number of degrees in a polynomial'
    NWINDOWS = 'number of spectral windows'




@dc.dataclass(slots=True)
class Model1(PreRTModelBase):
    """
        Variable deep abundance, fixed knee pressure and variable fractional scale height.
    """
    id: ClassVar[int] = 1

    deep_abundance: StateParam.using(slice(0,1), 'Deep abundance') # noqa: F722 F821
    frac_scale_height: StateParam.using(slice(1,2), 'Fractional scale height') # noqa: F722 F821

    atm_profile_type: ConstParam.using(AtmosphericProfileTypeEnum, 'Atmospheric profile type this model applies to') # noqa: F722 F821
    pknee: VarParam.using(float,'Knee pressure (atm)') # noqa: F722 F821

    def calculate(
        self,
        atm: "Atmosphere_0",
        atm_profile_type: AtmosphericProfileTypeEnum,
        atm_profile_idx: int | None,
        MakePlot=False,
    ) -> tuple["Atmosphere_0", np.ndarray]:
        """
            FUNCTION NAME : model1()

            DESCRIPTION :

                Variable deep abundance, fixed knee pressure and variable fractional scale height.    

            INPUTS :

                atm :: Python class defining the atmosphere
                ABU_DEEP :: Deep abundance
                FSH :: Fractional scale height
                PKNEE :: Knee pressure (atm)

            OPTIONAL INPUTS:

                MakePlot :: If True, a summary plot is generated

            OUTPUTS :

                atm :: Updated atmospheric class
                xmap(mparam,npro) :: Matrix of relating funtional derivatives to 
                                                 elements in state vector

            CALLING SEQUENCE:

                atm,xmap = model62(atm,p1,p2,p3,t0,alpha1,alpha2)

            MODIFICATION HISTORY : Juan Alday (18/12/2025)

        """

        PKNEE = self.pknee.v
        ABU_DEEP = self.deep_abundance.v
        FSH = self.frac_scale_height.v

        xfac = (1.0 - FSH) / FSH

        #d(xfac)/d(FSH)
        dxfac = -1.0 / FSH**2.

        #Finding the knee altitude
        pknee_pa = PKNEE * 101325.   #Calculating knee pressure in Pa
        isort = np.argsort(atm.P)
        p_sorted = atm.P[isort]
        h_sorted = atm.H[isort]
        hknee = np.interp(pknee_pa, p_sorted, h_sorted) #metres

        #Calculating the scale height
        R = const.R
        scale = R * atm.T / (atm.MOLWT * atm.GRAV)   #scale height (m)

        #Creating the new vertical profile and the functional derivatives
        xprof = np.zeros(atm.NP)
        xmap = np.zeros((2,atm.NP))   #Matrix with functional derivates of the parameters wrt the profiles
        jfsh = 0
        for j in range(atm.NP):

            #Above knee pressure
            if atm.P[j]>=pknee_pa:

                xprof[j] = ABU_DEEP
                xmap[0,j] = 1.0

            else:

                if jfsh == 0:
                    delh = atm.H[j] - hknee
                else:
                    delh = atm.H[j] - atm.H[j - 1]

                xprof[j]=xprof[j-1]*np.exp(-delh*xfac/scale[j])

                #Functional derivative of ABU_DEEP
                xmap[0,j] = xmap[0,j-1] * np.exp(-delh * xfac / scale[j])

                #Functional derivative of FSH
                xmap[1,j] = (
                    (-delh / scale[j])
                    * dxfac
                    * xprof[j-1]
                    * np.exp(-delh * xfac / scale[j])
                    + xmap[1,j-1]
                    * np.exp(-delh * xfac / scale[j])
                )

                jfsh = 1

                if xprof[j] < 1.0e-36:
                    xprof[j] = 1.0e-36

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
            atm.PARAH2(xprof)
        elif atm_profile_type == AtmosphericProfileTypeEnum.FRACTIONAL_CLOUD_COVERAGE:
            atm.FRAC(xprof)
        else:
            raise ValueError(f'{self.__class__.__name__} id {self.id} has unknown atmospheric profile type {atm_profile_type}')

        return atm, xmap

    @classmethod
    def from_apr_file(
        cls,
        f: IO,
        varident: np.ndarray[[3],int],
        npro: int,
        ngas: int,
        ndust: int,
        nlocations: int,
        runname: str,
        sxminfac: float,
        input_file_type: ArchNemesisFileTypeEnum,
    ) -> Self:
        s = f.readline().split()
        pknee = float(s[0])

        xvals_raw, xerrs_raw = cls.read_apr_value_error_pairs(f, 2)

        instance = cls.from_arrays(
            xvals_raw,
            xerrs_raw,
            cls.get_model_profile_type_enum_from_varident(varident, ngas, ndust),
            pknee,
        )

        if varident[0] != 0:
            instance.deep_abundance.log = True
            instance.frac_scale_height.log = True

        return instance

    @classmethod
    def from_bookmark(
        cls,
        variables: "Variables_0",
        varident: np.ndarray[[3],int],
        varparam: np.ndarray[["mparam"],float],
        ix: int,
        npro: int,
        ngas: int,
        ndust: int,
        nlocations: int,
    ) -> Self:
        pknee = varparam[0]
        xvals = np.zeros(2)
        xerrs = np.zeros(2)
        return cls.from_arrays(
            xvals,
            xerrs,
            cls.get_model_profile_type_enum_from_varident(varident, ngas, ndust),
            pknee,
        )