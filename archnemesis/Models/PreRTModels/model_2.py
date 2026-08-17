
from typing import TYPE_CHECKING, Self, IO, ClassVar
import dataclasses as dc

import numpy as np
import matplotlib.pyplot as plt

from ._base import PreRTModelBase
from ..param import StateParam, ConstParam


from archnemesis.enum import AtmosphericProfileTypeEnum, ArchNemesisFileTypeEnum

from ..log import _lgr  # noqa # Ignore if _lgr is not used


if TYPE_CHECKING:
    # NOTE: This is just here to make 'flake8' play nice with the type hints
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
class Model2(PreRTModelBase):
    """
        In this model, the atmospheric parameters are scaled using a single factor with
        respect to the vertical profiles in the reference atmosphere
    """
    id: ClassVar[int] = 2

    scaling_factor: StateParam.using(slice(0,1), 'Scaling factor applied to the reference profile') # noqa: F722 F821
    atm_profile_type: ConstParam.using(AtmosphericProfileTypeEnum, 'Atmospheric profile type this model applies to') # noqa: F722 F821

    def calculate(
        self,
        atm: "Atmosphere_0",
        atm_profile_type: AtmosphericProfileTypeEnum,
        atm_profile_idx: int | None,
        MakePlot=False,
    ):

        """
            FUNCTION NAME : model2()

            DESCRIPTION :

                Function defining the model parameterisation 2 in NEMESIS.
                In this model, the atmospheric parameters are scaled using a single factor with 
                respect to the vertical profiles in the reference atmosphere

            INPUTS :

                atm :: Python class defining the atmosphere

                atm_profile_type :: AtmosphericProfileTypeEnum
                    ENUM of atmospheric profile type we are altering.
                
                atm_profile_idx : int | None
                    Index of the atmospheric profile we are altering (or None if the profile type does not have multiples)

                scf :: Scaling factor

            OPTIONAL INPUTS:

                MakePlot :: If True, a summary plot is generated

            OUTPUTS :

                atm :: Updated atmospheric class
                xmap(1,npro) :: Matrix of relating funtional derivatives to 
                                                 elements in state vector

            CALLING SEQUENCE:

                atm,xmap = model2(atm,ipar,scf)

            MODIFICATION HISTORY : Juan Alday (29/03/2021)

        """

        xmap = np.zeros((1,atm.NP))
        scf = self.scaling_factor.v
        
        if atm_profile_type == AtmosphericProfileTypeEnum.GAS_VOLUME_MIXING_RATIO:
            xmap[0,:] = atm.VMR[:, atm_profile_idx]
            atm.VMR[:, atm_profile_idx] *= scf
        
        elif atm_profile_type == AtmosphericProfileTypeEnum.TEMPERATURE:
            xmap[0,:] = atm.T
            atm.T *= scf
        
        elif atm_profile_type == AtmosphericProfileTypeEnum.AEROSOL_DENSITY:
            xmap[0,:] = atm.DUST[:, atm_profile_idx]
            atm.DUST[:, atm_profile_idx] *= scf
        
        elif atm_profile_type == AtmosphericProfileTypeEnum.PARA_H2_FRACTION:
            xmap[0,:] = atm.PARAH2
            atm.PARAH2 *= scf
        
        elif atm_profile_type == AtmosphericProfileTypeEnum.FRACTIONAL_CLOUD_COVERAGE:
            xmap[0,:] = atm.FRAC
            atm.FRAC *= scf
        
        else:
            raise ValueError(f'{self.__class__.__name__} id {self.id} has unknown atmospheric profile type {atm_profile_type}')
        

        if MakePlot==True:
            fig,(ax1,ax2,ax3) = plt.subplots(1,3,figsize=(10,5))

            ax1.semilogx(atm.P/101325.,atm.H/1000.)
            ax2.plot(atm.T,atm.H/1000.)
            for i in range(atm.NVMR):
                ax3.semilogx(atm.VMR[:,i],atm.H/1000.)

            ax1.grid()
            ax2.grid()
            ax3.grid()
            ax1.set_xlabel('Pressure (atm)')
            ax1.set_ylabel('Altitude (km)')
            ax2.set_xlabel('Temperature (K)')
            ax2.set_ylabel('Altitude (km)')
            ax3.set_xlabel('Volume mixing ratio')
            ax3.set_ylabel('Altitude (km)')
            plt.tight_layout()
            plt.show()

        return atm,xmap


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
        xvals_raw, xerrs_raw = cls.read_apr_value_error_pairs(f, 1)
        instance = cls.from_arrays(
            xvals_raw,
            xerrs_raw,
            cls.get_model_profile_type_enum_from_varident(varident, ngas, ndust),
        )

        if varident[0] != 0:
            instance.scaling_factor.log = True

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
        xvals = np.zeros(1)
        xerrs = np.zeros(1)
        return cls.from_arrays(
            xvals,
            xerrs,
            cls.get_model_profile_type_enum_from_varident(varident, ngas, ndust),
        )


