
from typing import TYPE_CHECKING, Self, IO

import numpy as np
import matplotlib.pyplot as plt

from ._base import PreRTModelBase
from ..ModelParameter import ModelParameter
from archnemesis.enum import AtmosphericProfileTypeEnum

from ..log import _lgr  # noqa # Ignore if _lgr is not used


if TYPE_CHECKING:
    # NOTE: This is just here to make 'flake8' play nice with the type hints
    # the problem is that importing Variables_0 or ForwardModel_0 creates a circular import
    # this actually means that I should possibly redesign how those work to avoid circular imports
    # but that is outside the scope of what I want to accomplish here
    from archnemesis.Variables_0 import Variables_0
    from archnemesis.ForwardModel_0 import ForwardModel_0
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

class Model103(PreRTModelBase):
    """
        In this model, the atmospheric parameters of the telluric atmosphere are scaled 
        using a single factor in logscale
    """
    
    id : int = 103


    def __init__(
            self, 
            state_vector_start : int, 
            #   Index of the state vector where parameters from this model start
            
            n_state_vector_entries : int,
            #   Number of parameters for this model stored in the state vector
            
            atm_profile_type : AtmosphericProfileTypeEnum,
            #   ENUM that tells us what kind of atmospheric profile this model instance represents
        ):
        """
            Initialise an instance of the model.
        """
        super().__init__(state_vector_start, n_state_vector_entries, atm_profile_type)
        
        # Define sub-slices of the state vector that correspond to
        # parameters of the model.
        # NOTE: It is best to define these in the same order and with the
        # same names as they are saved to the state vector, and use the same
        # names and ordering when they are passed to the `self.calculate(...)` 
        # class method.
        self.parameters = (
            ModelParameter('scaling_factor', slice(0,1), 'Scaling factor applied to the reference profile, stored as a log in the state vector', 'PROFILE_TYPE'),
        )
        
        return


    @classmethod
    def calculate(
            cls, 
            telluric : "Telluric_0",
            #   Instance of Telluric_0 class we are operating upon
            
            atm_profile_type : AtmosphericProfileTypeEnum,
            #   ENUM of atmospheric profile type we are altering.
            
            atm_profile_idx : int | None,
            #   Index of the atmospheric profile we are altering (or None if the profile type does not have multiples)
            
            scf : float,
            #   scaling factor to be applied to the reference vertical profile
        ):

        """
            FUNCTION NAME : model103()

            DESCRIPTION :

                Function defining the model parameterisation 103 in NEMESIS.
                In this model, the atmospheric parameters of the telluric atmosphere are scaled 
                using a single factor in logscale with respect to the vertical profiles 
                in the reference telluric atmosphere

            INPUTS :

                telluric :: Python class defining the telluric atmosphere

                atm_profile_type :: AtmosphericProfileTypeEnum
                    ENUM of atmospheric profile type we are altering.
                
                atm_profile_idx : int | None
                    Index of the atmospheric profile we are altering (or None if the profile type does not have multiples)

                scf :: scaling factor

            OPTIONAL INPUTS:

                MakePlot :: If True, a summary plot is generated

            OUTPUTS :

                telluric :: Updated atmospheric class

            CALLING SEQUENCE:

                telluric = model103(telluric,ipar,scf)

            MODIFICATION HISTORY : Juan Alday (05/09/2026)

        """

        xmap = np.zeros((1,telluric.Atmosphere.NP))
        
        if atm_profile_type == AtmosphericProfileTypeEnum.GAS_VOLUME_MIXING_RATIO:
            xmap[0,:] = telluric.Atmosphere.VMR[:, atm_profile_idx]
            telluric.Atmosphere.VMR[:, atm_profile_idx] *= scf
        
        elif atm_profile_type == AtmosphericProfileTypeEnum.TEMPERATURE:
            xmap[0,:] = telluric.Atmosphere.T
            telluric.Atmosphere.T *= scf
        
        elif atm_profile_type == AtmosphericProfileTypeEnum.AEROSOL_DENSITY:
            xmap[0,:] = telluric.Atmosphere.DUST[:, atm_profile_idx]
            telluric.Atmosphere.DUST[:, atm_profile_idx] *= scf
        
        elif atm_profile_type == AtmosphericProfileTypeEnum.PARA_H2_FRACTION:
            xmap[0,:] = telluric.Atmosphere.PARAH2
            telluric.Atmosphere.PARAH2 *= scf
        
        elif atm_profile_type == AtmosphericProfileTypeEnum.FRACTIONAL_CLOUD_COVERAGE:
            xmap[0,:] = telluric.Atmosphere.FRAC
            telluric.Atmosphere.FRAC *= scf
        
        else:
            raise ValueError(f'{cls.__name__} id {cls.id} has unknown atmospheric profile type {atm_profile_type}')

        return telluric


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
        #**** model 103 - Exponential scaling factor of reference profile in telluric atmosphere *******
        #Read in scaling factor

        tmp = np.fromstring(f.readline().rsplit('!',1)[0], sep=' ',count=2,dtype='float') # Use "!" as comment character in *.apr files
        xfac = float(tmp[0])
        err = float(tmp[1])

        if xfac > 0.0:
            x0[ix] = np.log(xfac)
            lx[ix] = 1
            inum[ix] = 1
            sx[ix,ix] = ( err/xfac ) **2.
        else:
            raise ValueError('Error in read_apr_nemesis().  xfac must be > 0')

        ix = ix + 1

        model_classification = variables.classify_model_type_from_varident(varident, ngas, ndust)
        assert issubclass(cls, model_classification[0]), "Model base class must agree with the classification from Variables_0::classify_model_type_from_varident"

        return cls(ix_0, ix-ix_0, model_classification[1])


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
        _lgr.debug(f'Initialising model {cls.__name__} setup from bookmark')
        ix_0 = ix
        #**** model 102 - Exponential scaling factor of reference telluric profile *******
        if varident[2] != cls.id:
            raise ValueError('error in Model103.from_bookmark() :: wrong model id')

        ix = ix + 1

        model_classification = variables.classify_model_type_from_varident(varident, ngas, ndust)
        assert issubclass(cls, model_classification[0]), "Model base class must agree with the classification from Variables_0::classify_model_type_from_varident"

        return cls(ix_0, ix-ix_0, model_classification[1])

    def calculate_from_subprofretg(
            self,
            forward_model : "ForwardModel_0",
            ix : int,
            ipar : int,
            ivar : int,
            xmap : np.ndarray,
        ) -> None:
        #Model 103. Log scaling factor of telluric profile
        #***************************************************************
        telluric = forward_model.TelluricX
        atm_profile_type, atm_profile_idx = telluric.Atmosphere.ipar_to_atm_profile_type(ipar)
        
        telluric = self.calculate(
            telluric,
            atm_profile_type,
            atm_profile_idx,
            *self.get_parameter_values_from_state_vector(forward_model.Variables.XN, forward_model.Variables.LX)
        )
        
        forward_model.TelluricX = telluric
        
        return


