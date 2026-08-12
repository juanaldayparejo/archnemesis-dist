
from typing import TYPE_CHECKING, Self, IO, ClassVar
import dataclasses as dc

import numpy as np
#import matplotlib.pyplot as plt

from ..param import (
    StateParam, 
    #ConstParam, 
    #VarParam,
)

from ._base import PreRTModelBase

from archnemesis.enum import ArchNemesisFileTypeEnum

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
class Model999(PreRTModelBase):
    """
        In this model, the temperature of the surface is defined.
    """
    id : ClassVar[int] = 999 
    
    surface_temperature : StateParam.using(slice(0,1), "Surface temperature of planet", 'Kelvin') # noqa: F722 F821
    
    
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
            xerrs
        )
        
        instance.surface_temperature.log = False
        
        return instance
    
        
    def calculate(self, Surface):

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

        Surface.TSURF = self.surface_temperature.v

        return Surface


    

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

        return cls((0,0))

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
        forward_model.SurfaceX = self.calculate(forward_model.SurfaceX)











