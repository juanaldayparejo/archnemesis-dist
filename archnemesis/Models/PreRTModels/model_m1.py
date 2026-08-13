



from typing import TYPE_CHECKING, Self, IO, ClassVar
import dataclasses as dc

import numpy as np
#import matplotlib.pyplot as plt

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
class Modelm1(PreRTModelBase):
    """
    In this model, the dust density is modelled as continuous profiles
    in which each element of the state vector corresponds to the dust density 
    at each altitude level. The units are different when invoked via LEGACY or
    ARCHNEMESIS input files.
    
    LEGACY UNITS : particles cm^{-3}
    ARCHNEMESIS UNITS : particles gram^{-1}
    
    ## DETAILS ##
    
    "Normal" archnemesis units are 'particles cm^{-3}', and FORTRAN-nemesis units are 'particles gram^{-1}',
    this model swaps which one is being used depending upon the source of the input files.
    
    The unit being used is recorded in the 'Atmosphere_0.DUST_UNITS_FLAG' attribute. When reading ArchNemesis
    input (HDF5) files, this is set to `None`, when reading LEGACY input files it is set to a numpy array filled
    with `-1` values (for each dust profile).
    
    """
    
    id : ClassVar[int] = -1

    full_profile     : StateParam.using(slice(None), 'Every value for each level of the profile', 'PROFILE_TYPE') # noqa: F722 F821
    
    atm_profile_type   : ConstParam[AtmosphericProfileTypeEnum].using('Atmospheric profile type this model applies to') # noqa: F722 F821
    input_file_type    : ConstParam[ArchNemesisFileTypeEnum].using('Input file type we are using the "alternate" units of.') # noqa: F722 F821
    n_level            : ConstParam[int].using('Number of levels in the profile') # noqa: F722 F821
    pressure           : ConstParam[np.ndarray].using('Pressure at each entry of the profile') # noqa: F722 F821
    correlation_length : ConstParam[float].using('Correlation length of profile') # noqa: F722 F821
    
    
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
        
        instance.full_profile.log = instance.atm_profile_type.v != AtmosphericProfileTypeEnum.TEMPERATURE
        
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
            FUNCTION NAME : modelm1()

            DESCRIPTION :

                Function defining the model parameterisation -1 in NEMESIS.
                In this model, the aerosol profiles is modelled as a continuous profile in units
                of particles perModelm1 gram of atmosphere. Note that typical units of aerosol profiles in NEMESIS
                are in particles per gram of atmosphere

            INPUTS :

                atm :: Python class defining the atmosphere

                atm_profile_type :: AtmosphericProfileTypeEnum
                        ENUM of atmospheric profile type we are altering.
                    
                atm_profile_idx : int | None
                    Index of the atmospheric profile we are altering (or None if the profile type does not have multiples)

                xprof(npro) :: Atmospheric aerosol profile in particles/cm3

            OPTIONAL INPUTS:

                MakePlot :: If True, a summary plot is generated

            OUTPUTS :

                atm :: Updated atmospheric class
                xmap(npro,ngas+2+ncont,npro) :: Matrix of relating funtional derivatives to 
                                                    elements in state vector

            CALLING SEQUENCE:

                atm,xmap = modelm1(atm,ipar,xprof)

            MODIFICATION HISTORY : Juan Alday (29/03/2021)

        """
        xprof = self.full_profile.v
        npro = len(xprof)
        if npro!=atm.NP:
            raise ValueError('error in model -1 :: Number of levels in atmosphere does not match the passed profile')
        
        
        xmap = np.diag(xprof) if self.full_profile.log else np.diag(np.ones_like(xprof))
        
        if atm_profile_type == AtmosphericProfileTypeEnum.AEROSOL_DENSITY:
            temp = np.array(atm.DUST)
            temp[:,atm_profile_idx] = xprof
            atm.edit_DUST(temp)
            
            
            if self.input_file_type == ArchNemesisFileTypeEnum.LEGACY:
                if atm.DUST_UNITS_FLAG is not None:
                    atm.DUST_UNITS_FLAG[atm_profile_idx] = AerosolUnitEnum.NUMBER_DENSITY
            elif self.input_file_type == ArchNemesisFileTypeEnum.HDF5:
                if atm.DUST_UNITS_FLAG is None:
                    atm.DUST_UNITS_FLAG = np.full((atm.NDUST,), fill_value=AerosolUnitEnum.NUMBER_DENSITY, dtype=int)
                atm.DUST_UNITS_FLAG[atm_profile_idx] = AerosolUnitEnum.PARTICLES_PER_GRAM
            else:
                assert self.input_file_type != ArchNemesisFileTypeEnum.UNDEFINED, "Model0 must have a defined file type for input files."
        
        else:
            raise ValueError(f'error :: Model -1 is only compatible with aerosol profiles, not {atm_profile_type}')
            
        return atm, xmap





