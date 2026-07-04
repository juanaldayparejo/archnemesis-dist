
from typing import TYPE_CHECKING, Self, IO

import numpy as np

from ._base import PreRTModelBase

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

class Model556(PreRTModelBase):
    """
        In this model, we retrieve a scaling factor for the planetary radius
    """
    id : int = 556

    def __init__(
            self, 
            state_vector_start : int, 
            #   Index of the state vector where parameters from this model start
            
            n_state_vector_entries : int,
            #   Number of parameters for this model stored in the state vector

        ):
        """
            Initialise an instance of the model.
        """
        super().__init__(state_vector_start, n_state_vector_entries)

    @classmethod
    def calculate(cls, Atmosphere, radius_scaling_factor, MakePlot=False):

        """
            FUNCTION NAME : model556()

            DESCRIPTION :

                Function defining the model parameterisation 556 in NEMESIS.
                In this model, we retrieve a scaling factor for the planetary radius.

            INPUTS :

                Atmosphere :: Atmosphere class
                radius_scaling_factor :: Scaling factor for the planetary radius

            OPTIONAL INPUTS: None

            OUTPUTS :
                
                Atmosphere :: Updated Atmosphere class with recomputed pressure levels

            CALLING SEQUENCE:

                Atmosphere = model556(Atmosphere,radius_scaling_factor)

            MODIFICATION HISTORY : Juan Alday (15/02/2023)

        """
    
        _lgr.info(f'Calculating model 556 with radius_scaling_factor={radius_scaling_factor}')
        Atmosphere.PLANET_RADIUS = Atmosphere.PLANET_RADIUS * radius_scaling_factor
        Atmosphere.calc_grav()

        return Atmosphere

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
        #******** pressure at a given tangent height
        s = f.readline().split()
        radius_scaling_factor = float(s[0])
        radius_scaling_factor_err = float(s[1])

        x0[ix] = radius_scaling_factor
        lx[ix] = 0
        inum[ix] = 1

        sx[ix,ix] = (radius_scaling_factor_err)**2.
        #jpre = ix
    
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
            raise ValueError('error in Model556.from_bookmark() :: wrong model id')
        
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
        #Model 555. Retrieval of correction to planetary radius
        #***************************************************************

        radius_scaling_factor = forward_model.Variables.XN[ix]

        forward_model.AtmosphereX = self.calculate(forward_model.AtmosphereX,radius_scaling_factor)

        ix = ix + forward_model.Variables.NXVAR[ivar]


