

from typing import TYPE_CHECKING, IO, Self

import numpy as np

from ._base import PostRTModelBase

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

class Model667(PostRTModelBase):
    """
        In this model, the output spectrum is scaled using a dillusion factor to account
        for strong temperature gradients in exoplanets
    """
    id : int = 667


    @classmethod
    def calculate(cls, Spectrum,xfactor,MakePlot=False):

        """
            FUNCTION NAME : model667()

            DESCRIPTION :

                Function defining the model parameterisation 667 in NEMESIS.
                In this model, the output spectrum is scaled using a dillusion factor to account
                for strong temperature gradients in exoplanets

            INPUTS :

                Spectrum :: Modelled spectrum 
                xfactor :: Dillusion factor

            OPTIONAL INPUTS: None

            OUTPUTS :

                Spectrum :: Modelled spectrum scaled by the dillusion factor

            CALLING SEQUENCE:

                Spectrum = model667(Spectrum,xfactor)

            MODIFICATION HISTORY : Juan Alday (29/03/2021)

        """

        Spectrum = Spectrum * xfactor

        return Spectrum


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
        #******** dilution factor to account for thermal gradients thorughout exoplanet
        tmp = np.fromstring(f.readline().rsplit('!',1)[0], sep=' ',count=2,dtype='float') # Use "!" as comment character in *.apr files
        xfac = float(tmp[0])
        xfacerr = float(tmp[1])
        x0[ix] = xfac
        inum[ix] = 0 
        sx[ix,ix] = xfacerr**2.
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
        #******** dilution factor to account for thermal gradients thorughout exoplanet
        ix = ix + 1

        return cls(ix_0, ix-ix_0)

    def calculate_from_subspecret(
            self,
            forward_model : "ForwardModel_0",
            ix : int,
            ivar : int,
            SPECMOD : np.ndarray[['NCONV','NGEOM'],float],
            dSPECMOD : np.ndarray[['NCONV','NGEOM','NX'],float],
        ) -> None:
        #Model 667. Spectrum scaled by dilution factor to account for thermal gradients in planets
        #**********************************************************************************************

        xfactor = forward_model.Variables.XN[ix]
        spec = np.zeros(forward_model.SpectroscopyX.NWAVE)
        spec[:] = SPECMOD
        SPECMOD = self.calculate(SPECMOD,xfactor)
        dSPECMOD = dSPECMOD * xfactor
        dSPECMOD[:,ix] = spec[:]
        ix = ix + 1

