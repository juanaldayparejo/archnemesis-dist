
from typing import TYPE_CHECKING, Self, IO, ClassVar
import dataclasses as dc

import numpy as np
import matplotlib.pyplot as plt

from ._base import PreRTModelBase
from ..param import (
    StateParam, 
    ConstParam, 
    #VarParam,
)

from archnemesis.enum import AtmosphericProfileTypeEnum, ArchNemesisFileTypeEnum, AerosolUnitEnum

from ..log import _lgr  # noqa # Ignore if _lgr is not used


if TYPE_CHECKING:
    # NOTE: This is just here to make 'flake8' play nice with the type hints
    # the problem is that importing Variables_0 or ForwardModel_0 creates a circular import
    # this actually means that I should possibly redesign how those work to avoid circular imports
    # but that is outside the scope of what I want to accomplish here
    from archnemesis.Atmosphere_0 import Atmosphere_0

    nx = 'number of elements in state vector'



@dc.dataclass(slots=True)
class Model0(PreRTModelBase):
    """
    In this model, the atmospheric parameters are modelled as continuous profiles
    in which each element of the state vector corresponds to the atmospheric profile 
    at each altitude level
    """
    
    id : ClassVar[int] = 0

    full_profile     : StateParam.using(slice(None), 'Every value for each level of the profile', 'PROFILE_TYPE')
    
    atm_profile_type   : ConstParam[AtmosphericProfileTypeEnum].using('Atmospheric profile type this model applies to')
    input_file_type    : ConstParam[ArchNemesisFileTypeEnum].using('Input file type we are using the "alternate" units of.')
    n_level            : ConstParam[int].using('Number of levels in the profile')
    pressure           : ConstParam[np.ndarray].using('Pressure at each entry of the profile')
    correlation_length : ConstParam[float].using('Correlation length of profile')
    
    
    @classmethod
    def from_apr_file(
            cls,
            f : IO,
            varident : np.ndarray[[3],int],
            npro : int,
            ngas : int,
            ndust : int,
            nlocations : int,
            runname : str,
            sxminfac : float,
            input_file_type : ArchNemesisFileTypeEnum,
    ) -> Self:
        n_level, clen = cls.read_apr_entries(f, (int, float))
        assert n_level == npro, "Profiles must be on the same grid as .prf"
        pref, xvals, xerrs = cls.read_apr_entries(f, (float,float,float), n_level)
        assert np.all(pref >= 0), "Apriori file must be on pressure grid"
    
        instance = cls.from_arrays(
            xvals,
            xerrs,
            cls.get_model_profile_type_enum_from_varident(varident, ngas, ndust),
            input_file_type,
            n_level,
            pref,
            clen,
        )
        
        assert instance.atm_profile_type.v is not None, \
            f"{cls.__name__}[id={instance.id}] is only valid for atmospheric profiles"
        
        instance.full_profile.log = instance.atm_profile_type != AtmosphericProfileTypeEnum.TEMPERATURE
        
        return instance
    
    
    def push_to_covariance_matrix(
            self,
            sx : np.ndarray[["nx","nx"],float],
            sxminfac : float,
    ):
        self.push_to_covariance_matrix_with_correlation_length(
            sx,
            sxminfac,
            self.correlation_length.v,
        )
    

    def calculate(
            self, 
            atm : "Atmosphere_0",
            #   Instance of Atmosphere_0 class we are operating upon
            
            atm_profile_type : AtmosphericProfileTypeEnum,
            #   ENUM of atmospheric profile type we are altering.
            
            atm_profile_idx : int | None,
            #   Index of the atmospheric profile we are altering (or None if the profile type does not have multiples)
            
            MakePlot=False
    ) -> tuple["Atmosphere_0", np.ndarray]:
        """
            FUNCTION NAME : model0()

            DESCRIPTION :

                Function defining the model parameterisation 0 in NEMESIS.
                In this model, the atmospheric parameters are modelled as continuous profiles
                in which each element of the state vector corresponds to the atmospheric profile 
                at each altitude level

            INPUTS :

                atm :: Python class defining the atmosphere

                atm_profile_type :: AtmosphericProfileTypeEnum
                    ENUM of atmospheric profile type we are altering.
                
                atm_profile_idx : int | None
                    Index of the atmospheric profile we are altering (or None if the profile type does not have multiples)

                xprof(npro) :: Atmospheric profile

            OPTIONAL INPUTS:

                MakePlot :: If True, a summary plot is generated

            OUTPUTS :

                atm :: Updated atmospheric class
                xmap(npro,ngas+2+ncont,npro) :: Matrix of relating funtional derivatives to 
                                                model parameters.

            CALLING SEQUENCE:

                atm,xmap = model0(atm,ipar,xprof)

            MODIFICATION HISTORY : Juan Alday (29/03/2021)

        """
        
        xprof = self.full_profile.v
        _lgr.debug(f'Calculating {self.__name__} {atm=} {atm_profile_type=} {atm_profile_idx=} {xprof.shape=}')
        _lgr.debug(f'{xprof[:10]=}')
        
        

        npro = len(xprof)
        if npro!=atm.NP:
            raise ValueError('error in model 0 :: Number of levels in atmosphere does not match the passed profile')
        
        xmap = np.diag(np.diag(xprof)) if self.full_profile.log else np.diag(np.ones_like(xprof))
        
        if atm_profile_type == AtmosphericProfileTypeEnum.GAS_VOLUME_MIXING_RATIO:
            temp = np.array(atm.VMR)
            temp[:,atm_profile_idx] = xprof
            atm.edit_VMR(temp)
        
        elif atm_profile_type == AtmosphericProfileTypeEnum.TEMPERATURE:
            atm.edit_T(xprof)
        
        elif atm_profile_type == AtmosphericProfileTypeEnum.AEROSOL_DENSITY:
            temp = np.array(atm.DUST)
            temp[:,atm_profile_idx] = xprof
            atm.edit_DUST(temp)
            
            # Ensure atmosphere dust units are correct
            if self.input_file_type == ArchNemesisFileTypeEnum.LEGACY:
                if atm.DUST_UNITS_FLAG is None:
                    atm.DUST_UNITS_FLAG = np.full((atm.NDUST,), fill_value=AerosolUnitEnum.NUMBER_DENSITY, dtype=int)
                atm.DUST_UNITS_FLAG[atm_profile_idx] = AerosolUnitEnum.PARTICLES_PER_GRAM
            elif self.input_file_type == ArchNemesisFileTypeEnum.HDF5:
                if atm.DUST_UNITS_FLAG is not None:
                    atm.DUST_UNITS_FLAG[atm_profile_idx] = AerosolUnitEnum.NUMBER_DENSITY
            else:
                assert self.input_file_type != ArchNemesisFileTypeEnum.UNDEFINED, "Model0 must have a defined file type for input files."
            
        
        elif atm_profile_type == AtmosphericProfileTypeEnum.PARA_H2_FRACTION:
            atm.PARAH2(xprof)
        
        elif atm_profile_type == AtmosphericProfileTypeEnum.FRACTIONAL_CLOUD_COVERAGE:
            atm.FRAC(xprof)
        
        else:
            raise ValueError(f'{self.__name__} id {self.id} has unknown atmospheric profile type {atm_profile_type}')
        
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