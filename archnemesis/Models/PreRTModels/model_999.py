
from typing import TYPE_CHECKING, Self, IO

import numpy as np

from ._base import PreRTModelBase
from ..ModelParameter import ModelParameter

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

class Model999(PreRTModelBase):
    """
        In this model, the temperature of the surface is defined.
    """
    id : int = 999 
        
    def __init__(
            self, 
            state_vector_start : int = 0, 
            #   Index of the state vector where parameters from this model start
            
            n_state_vector_entries : int = 1,
            #   Number of parameters for this model stored in the state vector
        ):
        """
            Initialise an instance of the model.
        """
        super().__init__(state_vector_start, n_state_vector_entries)
        
        # Define sub-slices of the state vector that correspond to
        # parameters of the model
        self.parameters = (
            ModelParameter('surface temperature', slice(0,1), 'Surface Temperature','K'),
        )
        
        
    @classmethod
    def calculate(cls, Surface, tsurf):

        """
            FUNCTION NAME : model999()

            DESCRIPTION :

                Function defining the model parameterisation 999 in NEMESIS.
                In this model, we fit the surface temperature.

            INPUTS :

                Surface :: Python class defining the surface
                tsurf :: Surface temperature (K)

            OPTIONAL INPUTS: none

            OUTPUTS :

                Surface :: Updated measurement class with the surface temperature

            CALLING SEQUENCE:

                Surface = model999(Surface,tsurf)

            MODIFICATION HISTORY : Juan Alday (25/05/2025)

        """

        Surface.TSURF = tsurf

        return Surface


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
        #******** model for retrieving the Surface temperature

        #Read the surface temperature and its uncertainty
        s = f.readline().split()
        tsurf = float(s[0])     #K
        tsurf_err = float(s[1]) #K

        #Filling the state vector and a priori covariance matrix with the surface temperature
        lx[ix] = 0   #linear scale
        x0[ix] = tsurf
        sx[ix,ix] = (tsurf_err)**2.
        inum[ix] = 0  #analytical gradient

        ix = ix + 1

        return cls(ix_0, ix-ix_0)

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
        
        if varident[2] != cls.id:
            raise ValueError('error in Model999.from_bookmark() :: wrong model id')
        
        ix_0 = ix
        ix = ix + 1

        return cls(ix_0, ix-ix_0)

    def calculate_from_subprofretg(
            self,
            forward_model : "ForwardModel_0",
            ix : int,
            ipar : int,
            ivar : int,
            xmap : np.ndarray,
        ) -> None:
        #Model 999. Retrieval of surface temperature
        #***************************************************************

        tsurf = forward_model.Variables.XN[ix]

        forward_model.SurfaceX = self.calculate(forward_model.SurfaceX,tsurf)

        #ipar = -1
        ix = ix + forward_model.Variables.NXVAR[ivar]











