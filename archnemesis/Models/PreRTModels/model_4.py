



from typing import TYPE_CHECKING, Self, IO, ClassVar
import dataclasses as dc

import numpy as np


from ._base import PreRTModelBase
from ..param import StateParam, ConstParam

import archnemesis.Data.constants as const
from archnemesis.enum import AtmosphericProfileTypeEnum, ArchNemesisFileTypeEnum

from ..log import _lgr  # noqa # Ignore if _lgr is not used


if TYPE_CHECKING:
    # NOTE: This is just here to make 'flake8' play nice with the type hints
    # the problem is that importing Variables_0 or ForwardModel_0 creates a circular import
    # this actually means that I should possibly redesign how those work to avoid circular imports
    # but that is outside the scope of what I want to accomplish here
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
class Model4(PreRTModelBase):
    """
        Variable deep abundance, knee pressure and fractional scale height. 
    """
    id: ClassVar[int] = 4

    deep_abundance: StateParam.using(slice(0,1), 'Deep abundance') # noqa: F722 F821
    frac_scale_height: StateParam.using(slice(1,2), 'Fractional scale height') # noqa: F722 F821
    pknee: StateParam.using(slice(2,3), 'Knee pressure (atm)') # noqa: F722 F821
    atm_profile_type: ConstParam[AtmosphericProfileTypeEnum].using('Atmospheric profile type this model applies to') # noqa: F722 F821

    def calculate(
        self,
        atm: "Atmosphere_0",
        #   Instance of Atmosphere_0 class we are operating upon

        atm_profile_type: AtmosphericProfileTypeEnum,
        #   ENUM of atmospheric profile type we are altering.

        atm_profile_idx: int | None,
        #   Index of the atmospheric profile we are altering (or None if the profile type does not have multiples)

        MakePlot=False,
    ) -> tuple["Atmosphere_0", np.ndarray]:

        """
            FUNCTION NAME : model4()

            DESCRIPTION :

                Variable deep abundance, knee pressure and fractional scale height.    

            INPUTS :

                atm :: Python class defining the atmosphere
                PKNEE :: Knee pressure (atm)
                ABU_DEEP :: Deep abundance
                FSH :: Fractional scale height
                
            OPTIONAL INPUTS:

                MakePlot :: If True, a summary plot is generated

            OUTPUTS :

                atm :: Updated atmospheric class
                xmap(mparam,npro) :: Matrix of relating funtional derivatives to 
                                                 the model parameters

            MODIFICATION HISTORY : Juan Alday (03/07/2026)

        """        
        
        FSH = self.frac_scale_height.v
        ABU_DEEP = self.deep_abundance.v
        PKNEE = self.pknee.v

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
        xmap = np.zeros((3,atm.NP))   #Matrix with functional derivates
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

                #Functional derivative of PKNEE
                if jfsh == 0:
                    xmap[2,j] = (-xfac / (PKNEE)) * xprof[j-1] * np.exp(-delh * xfac / scale[j])

                xmap[2,j] = xmap[2,j] + xmap[2,j-1] * np.exp(-delh * xfac / scale[j])

                jfsh = 1

                if xprof[j] < 1.0e-36:
                    xprof[j] = 1.0e-36


        #Updating atmosphere class
        # If any state parameters are stored in log-space, the functional
        # derivatives must be multiplied by the parameter value (matching
        # the behavior used in Model0).
        for i, sp in enumerate(self.iter_stateparam_objs()):
            if sp.log:
                xmap[i, :] = xmap[i, :] * sp.v

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
        varident: np.ndarray[[3], int],
        npro: int,
        ngas: int,
        ndust: int,
        nlocations: int,
        runname: str,
        sxminfac: float,
        input_file_type: ArchNemesisFileTypeEnum,
    ) -> Self:
        xvals_raw, xerrs_raw = cls.read_apr_value_error_pairs(f, 3)
        instance = cls.from_arrays(
            xvals_raw,
            xerrs_raw,
            cls.get_model_profile_type_enum_from_varident(varident, ngas, ndust),
        )

        # In the legacy code, frac_scale_height and pknee were stored in log-space
        # and deep abundance was stored in log-space for non-temperature profiles.
        instance.frac_scale_height.log = True
        instance.pknee.log = True
        if varident[0] != 0:
            instance.deep_abundance.log = True

        return instance

    @classmethod
    def from_bookmark(
        cls,
        variables: "Variables_0",
        varident: np.ndarray[[3], int],
        varparam: np.ndarray[["mparam"], float],
        ix: int,
        npro: int,
        ngas: int,
        ndust: int,
        nlocations: int,
    ) -> Self:
        xvals = np.zeros(3)
        xerrs = np.zeros(3)
        return cls.from_arrays(
            xvals,
            xerrs,
            cls.get_model_profile_type_enum_from_varident(varident, ngas, ndust),
        )

    