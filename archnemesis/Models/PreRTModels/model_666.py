
from typing import TYPE_CHECKING, Self, IO

import numpy as np
import matplotlib.pyplot as plt

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

class Model666(PreRTModelBase):
    """
        In this model, we retrieve the pressure at a given tangent height.
    """
    id : int = 666

    def __init__(
            self, 
            state_vector_start : int, 
            #   Index of the state vector where parameters from this model start
            
            n_state_vector_entries : int,
            #   Number of parameters for this model stored in the state vector
            
            htan : float,
            #   Tangent height (km) at which the pressure is retrieved
        ):
        """
            Initialise an instance of the model.
        """
        super().__init__(state_vector_start, n_state_vector_entries, htan)
        
        # Define sub-slices of the state vector that correspond to
        # parameters of the model
        self.parameters = (
            ModelParameter('PTAN', slice(0,1), 'Pressure (at tangent height = '+str(htan)+' km)', 'atm'),
        )
        self.htan = htan

    @classmethod
    def calculate(cls, Atmosphere, htan, ptan, MakePlot=False):

        """
            FUNCTION NAME : model666()

            DESCRIPTION :

                Function defining the model parameterisation 666 in NEMESIS.
                In this model, we retrieve the pressure at a given tangent height.

            INPUTS :

                Atmosphere :: Atmosphere class
                htan :: Tangent height (km)
                ptan :: Pressure at tangent height (atm)

            OPTIONAL INPUTS: None

            OUTPUTS :
                
                Atmosphere :: Updated Atmosphere class with recomputed pressure levels

            CALLING SEQUENCE:

                Atmosphere = model666(Atmosphere,htan,ptan)

            MODIFICATION HISTORY : Juan Alday (15/02/2023)

        """

        hpre = Atmosphere.H
        ppre = Atmosphere.P
    
        _lgr.info(f'Calculating model 666 with htan={htan} km and ptan={ptan} atm')

        Atmosphere.adjust_hydrostatP(htan*1.0e3,ptan*101325.)

        if MakePlot==True:

            fig,ax1 = plt.subplots(1,1,figsize=(3,4))
            ax1.plot(ppre,hpre/1.0e3,label='Uncorrected')
            ax1.plot(Atmosphere.P,Atmosphere.H/1.0e3,label='Corrected')
            ax1.legend()
            ax1.set_xlabel('Pressure (Pa)')
            ax1.set_ylabel('Altitude (km)')
            ax1.set_xscale('log')
            plt.tight_layout()

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
        htan = float(s[0])  #Tangent height (km)
        s = f.readline().split()
        ptan = float(s[0])
        ptanerr = float(s[1])

        varparam[0] = htan

        if ptan>0.0:
            x0[ix] = np.log(ptan)
            lx[ix] = 1
            inum[ix] = 1
        else:
            raise ValueError('error in read_apr_nemesis() :: pressure must be > 0')
    
        sx[ix,ix] = (ptanerr/ptan)**2.
        #jpre = ix
    
        ix = ix + 1

        return cls(ix_0, ix-ix_0, htan)

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
            raise ValueError('error in Model666.from_bookmark() :: wrong model id')
        
        ix_0 = ix
        ix = ix + 1
        htan = varparam[0]

        return cls(ix_0, ix-ix_0, htan)

    def calculate_from_subprofretg(
            self,
            forward_model : "ForwardModel_0",
            ix : int,
            ipar : int,
            ivar : int,
            xmap : np.ndarray,
        ) -> None:
        #Model 666. Retrieval of pressure at a given tangent height
        #***************************************************************

        ptan = np.exp(forward_model.Variables.XN[ix])

        forward_model.AtmosphereX = self.calculate(forward_model.AtmosphereX,self.htan,ptan)

        #ipar = -1
        ix = ix + forward_model.Variables.NXVAR[ivar]


