
from typing import TYPE_CHECKING, Self, IO, ClassVar
import dataclasses as dc

import numpy as np

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
class Model556(PreRTModelBase):
    """
        In this model, we retrieve a scaling factor for the planetary radius
    """
    id : ClassVar[int] = 556
    
    radius_scaling_factor : StateParam.using(slice(0,1), 'Scaling factor for planetary radius', '', num_diff=True) # noqa: F722 F821

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
        
        return instance


    def calculate(self, Atmosphere, MakePlot=False):

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
        radius_scaling_factor = self.radius_scaling_factor.v
    
        _lgr.info(f'Calculating model 556 with radius_scaling_factor={radius_scaling_factor}')
        Atmosphere.PLANET_RADIUS = Atmosphere.PLANET_RADIUS * radius_scaling_factor
        Atmosphere.calc_grav()

        return Atmosphere

    

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
        return cls((0,0))

    def calculate_from_subprofretg(
            self,
            forward_model : "ForwardModel_0",
            ix : int,
            ipar : int,
            ivar : int,
            xmap : np.ndarray,
        ) -> None:
        self.pull_from_state_vector(forward_model.Variables.XN, forward_model.Variables.LX)
        forward_model.AtmosphereX = self.calculate(forward_model.AtmosphereX)







