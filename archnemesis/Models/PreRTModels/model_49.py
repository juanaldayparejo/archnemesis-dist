
from typing import TYPE_CHECKING, Self, IO

import numpy as np

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

class Model49(PreRTModelBase):
    """
        In this model, the profile is scaled using a single factor with 
        respect to a reference profile.
    """
    id : int = 49

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
            ModelParameter('scale', slice(0,1), 'Factor to scale reference profile by', 'NUMBER'),
        )
        
        return

    @classmethod
    def calculate(
            cls, 
            atm : "Atmosphere_0", 
            ipar : int, 
            scale : float, 
            ref_gas : int, 
            ref_iso : int,
    ) -> tuple["Atmosphere_0", np.ndarray]:
        """
            FUNCTION NAME : Model49.calculate()

            DESCRIPTION :

                Function defining the model parameterisation 51 (49 in NEMESIS).
                In this model, the profile is scaled using a single factor with 
                respect to a reference profile.

            INPUTS :

                atm :: Python class defining the atmosphere

                ipar :: Atmospheric parameter to be changed
                        (0 to NVMR-1) :: Gas VMR
                        (NVMR) :: Temperature
                        (NVMR+1 to NVMR+NDUST-1) :: Aerosol density
                        (NVMR+NDUST) :: Para-H2
                        (NVMR+NDUST+1) :: Fractional cloud coverage

                scale :: Scaling factor
                ref_gas :: Reference gas
                ref_iso :: Reference isotope

            OUTPUTS :

                atm :: Updated atmospheric class
                xmap(1,ngas+2+ncont,npro) :: Matrix of relating derivatives to 
                                                 elements in state vector

            CALLING SEQUENCE:

                atm,xmap = Model49.calculate(atm,ipar,scf)

            MODIFICATION HISTORY : Juan Alday (29/03/2021)

        """
        npar = atm.NVMR+2+atm.NDUST

        iref_vmr = np.where((atm.ID == ref_gas)&(atm.ISO == ref_iso))[0][0]
        x1 = np.zeros(atm.NP)
        xref = np.zeros(atm.NP)

        xref[:] = atm.VMR[:,iref_vmr]
        x1[:] = xref * scale
        atm.VMR[:,ipar] = x1

        xmap = np.zeros([1,npar,atm.NP])

        xmap[0,ipar,:] = xref[:]

        return atm,xmap


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
        #********* multiple of different profile ************************
        prof = np.fromstring(f.readline().rsplit('!',1)[0], sep=' ',count=2,dtype='int') # Use "!" as comment character in *.apr files
        profgas = prof[0]
        profiso = prof[1]
        tmp = np.fromstring(f.readline().rsplit('!',1)[0], sep=' ',count=2,dtype='float') # Use "!" as comment character in *.apr files
        scale = tmp[0]
        escale = tmp[1]

        varparam[1] = profgas
        varparam[2] = profiso
        x0[ix] = np.log(scale)
        lx[ix] = 1
        err = escale/scale
        sx[ix,ix] = err**2.

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
        ix_0 = ix
        #********* multiple of different profile ************************
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
        #Model 51. Scaling of a reference profile
        #***************************************************************                
        scale_gas, scale_iso = forward_model.Variables.VARPARAM[ivar,1:3]
        
        forward_model.AtmosphereX, xmap1 = self.calculate(
            forward_model.AtmosphereX,
            ipar,
            *self.get_parameter_values_from_state_vector(forward_model.Variables.XN, forward_model.Variables.LX),
            scale_gas,
            scale_iso
        )
        
        xmap[self.state_vector_slice, ipar, 0:forward_model.AtmosphereX.NP] = xmap1



