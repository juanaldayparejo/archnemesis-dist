
from typing import TYPE_CHECKING, Self, IO

import numpy as np
import matplotlib.pyplot as plt

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

class Model777(PreRTModelBase):
    """
        In this model, we apply a correction to the tangent heights listed on the 
        Measurement class
    """
    id : int = 777


    @classmethod
    def calculate(cls, Measurement,hcorr,MakePlot=False):

        """
            FUNCTION NAME : model777()

            DESCRIPTION :

                Function defining the model parameterisation 777 in NEMESIS.
                In this model, we apply a correction to the tangent heights listed on the 
                Measurement class

            INPUTS :

                Measurement :: Measurement class
                hcorr :: Correction to the tangent heights (km)

            OPTIONAL INPUTS: None

            OUTPUTS :

                Measurement :: Updated Measurement class with corrected tangent heights

            CALLING SEQUENCE:

                Measurement = model777(Measurement,hcorr)

            MODIFICATION HISTORY : Juan Alday (15/02/2023)

        """

        #Getting the tangent heights
        tanhe = np.zeros(Measurement.NGEOM)
        tanhe[:] = Measurement.TANHE[:,0]

        #Correcting tangent heights
        tanhe_new = tanhe + hcorr

        #Updating Measurement class
        Measurement.TANHE[:,0] = tanhe_new

        if MakePlot==True:

            fig,ax1 = plt.subplots(1,1,figsize=(3,4))
            ax1.scatter(np.arange(0,Measurement.NGEOM,1),tanhe,label='Uncorrected')
            ax1.scatter(np.arange(0,Measurement.NGEOM,1),Measurement.TANHE[:,0],label='Corrected')
            ax1.set_xlabel('Geometry #')
            ax1.set_ylabel('Tangent height (km)')
            plt.tight_layout()

        return Measurement


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
        #******** tangent height correction
        s = f.readline().split()
        hcorr = float(s[0])
        herr = float(s[1])

        x0[ix] = hcorr
        sx[ix,ix] = herr**2.
        inum[ix] = 1

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
        ix_0 = ix
        #******** tangent height correction
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
        #Model 777. Retrieval of tangent height corrections
        #***************************************************************

        hcorr = forward_model.Variables.XN[ix]

        forward_model.MeasurementX = self.calculate(forward_model.MeasurementX,hcorr)

        #ipar = -1
        ix = ix + forward_model.Variables.NXVAR[ivar]


