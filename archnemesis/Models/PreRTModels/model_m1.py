
from typing import TYPE_CHECKING, Self, IO

import numpy as np

from ._base import PreRTModelBase
from .model_0 import Model0
from ..ModelParameter import ModelParameter

from archnemesis.enum import AtmosphericProfileTypeEnum

import logging
_lgr = logging.getLogger(__name__)
#_lgr.setLevel(logging.DEBUG)
_lgr.setLevel(logging.INFO)


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

class Modelm1(PreRTModelBase):
    """
    In this model, the aerosol profiles is modelled as a continuous profile in units
    of particles per gram of atmosphere. Note that typical units of aerosol profiles in NEMESIS
    are in particles per gram of atmosphere
    """
    
    id : int = -1

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
            ModelParameter('full_profile', slice(None), 'Every value for each level of the profile', 'PROFILE_TYPE'),
        )
        
        return

    @classmethod
    def calculate(
            cls, 
            atm : "Atmosphere_0",
            #   Instance of Atmosphere_0 class we are operating upon
            
            atm_profile_type : AtmosphericProfileTypeEnum,
            #   ENUM of atmospheric profile type we are altering.
            
            atm_profile_idx : int | None,
            #   Index of the atmospheric profile we are altering (or None if the profile type does not have multiples)
            
            xprof : np.ndarray[['mparam'],float],
            #   Full profile, this model defines every value for each profile level. Has been unlogged as required
            
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
        
        npro = len(xprof)
        if npro!=atm.NP:
            raise ValueError('error in model -1 :: Number of levels in atmosphere does not match and profile')
            
        if atm_profile_type == AtmosphericProfileTypeEnum.AEROSOL_DENSITY:
            temp = np.array(atm.DUST)
            temp[:,atm_profile_idx] = xprof
            atm.edit_DUST(temp)
            xmap = np.diag(xprof)
        
        else:
            raise ValueError(f'error :: Model -1 is only compatible with aerosol profiles, not {atm_profile_type}')
            
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
        #* continuous cloud, but cloud retrieved as particles/cm3 rather than
        #* particles per gram to decouple it from pressure.
        #********* continuous particles/cm3 profile ************************
        ix_0 = ix
        
        if varident[0] >= 0:
            raise ValueError('error in read_apr_nemesis :: model -1 type is only for use with aerosols')

        s = f.readline().split()
        
        with open(s[0], 'r') as f1:
            tmp = np.fromfile(f1,sep=' ',count=2,dtype='float')
            
            nlevel = int(tmp[0])
            if nlevel != npro:
                raise ValueError('profiles must be listed on same grid as .prf')
            
            clen = float(tmp[1])
            pref = np.zeros([nlevel])
            ref = np.zeros([nlevel])
            eref = np.zeros([nlevel])
            
            for j in range(nlevel):
                tmp = np.fromfile(f1,sep=' ',count=3,dtype='float')
                pref[j] = float(tmp[0])
                ref[j] = float(tmp[1])
                eref[j] = float(tmp[2])

                lx[ix+j] = 1
                x0[ix+j] = np.log(ref[j])
                sx[ix+j,ix+j] = ( eref[j]/ref[j]  )**2.

        #Calculating correlation between levels in continuous profile
        for j in range(nlevel):
            for k in range(nlevel):
                if pref[j] < 0.0:
                    raise ValueError('Error in read_apr_nemesis().  A priori file must be on pressure grid')

                delp = np.log(pref[k])-np.log(pref[j])
                arg = abs(delp/clen)
                xfac = np.exp(-arg)
                if xfac >= sxminfac:
                    sx[ix+j,ix+k] = np.sqrt(sx[ix+j,ix+j]*sx[ix+k,ix+k])*xfac
                    sx[ix+k,ix+j] = sx[ix+j,ix+k]
        ix = ix + nlevel

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
        #* continuous cloud, but cloud retrieved as particles/cm3 rather than
        #* particles per gram to decouple it from pressure.
        #********* continuous particles/cm3 profile ************************
        ix_0 = ix
        
        if varident[0] >= 0:
            raise ValueError('error in read_apr_nemesis :: model -1 type is only for use with aerosols')
        ix = ix + npro

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
        #Model -1. Continuous aerosol profile in particles cm-3
        #***************************************************************
        
        atm = forward_model.AtmosphereX
        atm_profile_type, atm_profile_idx = atm.ipar_to_atm_profile_type(ipar)
        
        if atm_profile_type == AtmosphericProfileTypeEnum.AEROSOL_DENSITY:
            calculate_fn = lambda *args, **kwargs: Model0.calculate(*args, **kwargs)
        else:
            calculate_fn = lambda *args, **kwargs: self.calculate(*args, **kwargs)
        
        atm, xmap1 = calculate_fn(
            atm,
            atm_profile_type,
            atm_profile_idx,
            *self.get_parameter_values_from_state_vector(forward_model.Variables.XN, forward_model.Variables.LX)
        )
        
        forward_model.AtmosphereX = atm
        xmap[self.state_vector_slice, ipar, 0:atm.NP] = xmap1
        
        return


    def patch_from_subprofretg(
            self,
            forward_model : "ForwardModel_0",
            ix : int,
            ipar : int,
            ivar : int,
            xmap : np.ndarray,
        ) -> None:
        #Model -1. Continuous aerosol profile in particles cm-3
        #***************************************************************
        atm = forward_model.AtmosphereX
        atm_profile_type, atm_profile_idx = atm.ipar_to_atm_profile_type(ipar)
        
        atm, xmap1 = self.calculate(
            atm,
            atm_profile_type,
            atm_profile_idx,
            *self.get_parameter_values_from_state_vector(forward_model.Variables.XN, forward_model.Variables.LX)
        )
        
        forward_model.AtmosphereX = atm
        xmap[self.state_vector_slice, ipar, 0:atm.NP] = xmap1
        
        return


