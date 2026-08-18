
from typing import TYPE_CHECKING, Self, IO, ClassVar
import dataclasses as dc

import numpy as np

from ..param import (
    StateParam, 
    #ConstParam, 
    #VarParam,
)

from archnemesis.enum import ArchNemesisFileTypeEnum

from ._base import PreRTModelBase

from ..log import _lgr  # noqa # Ignore if _lgr is not used


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

@dc.dataclass
class Model447(PreRTModelBase):
    """
        In this model, we fit the Doppler shift of the observation. Currently this Doppler shift
        is common to all geometries, but in the future it will be updated so that each measurement
        can have a different Doppler velocity (in order to retrieve wind speeds).
    """
    id : ClassVar[int] = 447

    v_dopper : StateParam.using('Doppler shift of observation', 'km/s') # noqa: F722 F821

    def calculate(self, Measurement,v_doppler):

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
    def from_apr_file(
            cls,
            f: IO,
            varident: np.ndarray[[3], int],
            npro: int,
            ngas: int,
            ndust: int,
            nlocations: int,
            runname: str,
            sxminfac: float,
            input_file_type: ArchNemesisFileTypeEnum,
    ) -> Self:
        xvals, xerrs = cls.read_apr_value_error_pairs(f, 1)
        
        instance = cls.from_arrays(
            xvals,
            xerrs,
        )
        
        instance.v_dopper.log = False
        
        return instance
    
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
        instance = cls.from_arrays(
            np.zeros((1,)),
            np.zeros((1,)),
        )
        
        instance.v_dopper.log = False

        return instance

    def calculate_from_subprofretg(
            self,
            forward_model : "ForwardModel_0",
            ix : int,
            ipar : int,
            ivar : int,
            xmap : np.ndarray,
        ) -> None:
        raise NotImplementedError


