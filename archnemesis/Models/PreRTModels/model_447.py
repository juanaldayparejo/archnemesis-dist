
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

class Model447(PreRTModelBase):
    """
        In this model, we fit the Doppler shift of the observation. Currently this Doppler shift
        is common to all geometries, but in the future it will be updated so that each measurement
        can have a different Doppler velocity (in order to retrieve wind speeds).
    """
    id : int = 447


    @classmethod
    def calculate(cls, Measurement,v_doppler):

        """
            FUNCTION NAME : model447()

            DESCRIPTION :

                Function defining the model parameterisation 447 in NEMESIS.
                In this model, we fit the Doppler shift of the observation. Currently this Doppler shift
                is common to all geometries, but in the future it will be updated so that each measurement
                can have a different Doppler velocity (in order to retrieve wind speeds).

            INPUTS :

                Measurement :: Python class defining the measurement
                v_doppler :: Doppler velocity (km/s)

            OPTIONAL INPUTS: none

            OUTPUTS :

                Measurement :: Updated measurement class with the correct Doppler velocity

            CALLING SEQUENCE:

                Measurement = model447(Measurement,v_doppler)

            MODIFICATION HISTORY : Juan Alday (25/07/2023)

        """

        Measurement.V_DOPPLER = v_doppler

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
        #******** model for retrieving the Doppler shift

        #Read the Doppler velocity and its uncertainty
        s = f.readline().split()
        v_doppler = float(s[0])     #km/s
        v_doppler_err = float(s[1]) #km/s

        #Filling the state vector and a priori covariance matrix with the doppler velocity
        lx[ix] = 0
        x0[ix] = v_doppler
        sx[ix,ix] = (v_doppler_err)**2.
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
        #******** model for retrieving the Doppler shift
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
        raise NotImplementedError


